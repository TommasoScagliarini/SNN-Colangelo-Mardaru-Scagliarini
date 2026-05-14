"""
PPO agent for Spiking Neural Networks, adapted from skrl's PPO_RNN.

The SNN membrane potential acts as an implicit recurrent state, removing the
need for windowed BPTT and sequence sampling required by standard recurrent PPO.
The membrane is carried across steps within an episode and zeroed at termination.
"""

import copy
import itertools
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import gymnasium

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveRL

from src.utils.entropy_scheduler import LinearEntropyDecay


PPO_SNN_DEFAULT_CONFIG = {
    "rollouts": 16,
    "learning_epochs": 8,
    "mini_batches": 2,

    "discount_factor": 0.99,
    "lambda": 0.95,

    "learning_rate": 1e-3,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},

    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},
    "value_preprocessor": None,
    "value_preprocessor_kwargs": {},

    "random_timesteps": 0,
    "learning_starts": 0,

    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": False,

    "entropy_loss_scale": 0.0,
    "entropy_loss_scale_end": 0.0,
    "value_loss_scale": 1.0,

    "kl_threshold": 0,
    "rewards_shaper": None,

    "net_activity": False,
    "net_actions": False,
    "net_decay_threshold": False,

    "experiment": {
        "directory": "",
        "experiment_name": "",
        "write_interval": 250,
        "checkpoint_interval": 1000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    },
}


class PPO_SNN(Agent):

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space=None,
        action_space=None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        num_envs: int = 1,
    ):
        _cfg = copy.deepcopy(PPO_SNN_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        self.policy = self.models.get("policy", None)
        self.value  = self.models.get("value",  None)
        self.num_envs = num_envs

        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"]  = self.value

        self._learning_epochs       = self.cfg["learning_epochs"]
        self._mini_batches          = self.cfg["mini_batches"]
        self._rollouts              = self.cfg["rollouts"]
        self._rollout               = 0

        self._grad_norm_clip        = self.cfg["grad_norm_clip"]
        self._ratio_clip            = self.cfg["ratio_clip"]
        self._value_clip            = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale      = self.cfg["value_loss_scale"]
        self._entropy_loss_scale    = self.cfg["entropy_loss_scale"]

        self._kl_threshold          = self.cfg["kl_threshold"]
        self._learning_rate         = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor    = self.cfg["state_preprocessor"]
        self._value_preprocessor    = self.cfg["value_preprocessor"]

        self._discount_factor       = self.cfg["discount_factor"]
        self._lambda                = self.cfg["lambda"]
        self._random_timesteps      = self.cfg["random_timesteps"]
        self._learning_starts       = self.cfg["learning_starts"]
        self._rewards_shaper        = self.cfg["rewards_shaper"]

        # SNN logging
        self._log_activity          = self.cfg.get("net_activity", False)
        self._log_counter           = 0
        self._log_interval          = 100

        # Entropy decay schedule
        self._entropy_scheduler = LinearEntropyDecay(
            start=self.cfg.get("entropy_loss_scale", 0.0),
            end=self.cfg.get("entropy_loss_scale_end", 0.0),
            total_steps=self.cfg.get("timesteps", 800_000),
        )

        # Single optimizer when policy and value share the same module.
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(
                    self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()),
                    lr=self._learning_rate)

            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # Preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(
                **self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        if self.memory is not None:
            self.memory.create_tensor(name="states",     size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions",    size=self.action_space,      dtype=torch.float32)
            self.memory.create_tensor(name="rewards",    size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1,                      dtype=torch.bool)
            self.memory.create_tensor(name="log_prob",   size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="values",     size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="returns",    size=1,                      dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1,                      dtype=torch.float32)
            self._tensors_names = ["states", "actions", "terminated", "log_prob",
                                   "values", "returns", "advantages"]

        # RNN state, managed manually.
        self._rnn = False
        self._rnn_tensors_names   = []
        self._rnn_final_states    = {"policy": [], "value": []}
        self._rnn_initial_states  = {"policy": [], "value": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # Policy RNN slots.
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            if self.memory is not None:
                self.memory.create_tensor(name=f"rnn_policy_{i}",
                                          size=(size[0], size[2]),
                                          dtype=torch.float32,
                                          keep_dimensions=True)
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            self._rnn_initial_states["policy"].append(
                torch.zeros(size, dtype=torch.float32, device=self.device))

        # Value RNN slots.
        if self.value is not None:
            if self.policy is self.value:
                # Shared module: share the RNN states too.
                self._rnn_initial_states["value"] = self._rnn_initial_states["policy"]
            else:
                for i, size in enumerate(self.value.get_specification().get("rnn", {}).get("sizes", [])):
                    self._rnn = True
                    if self.memory is not None:
                        self.memory.create_tensor(name=f"rnn_value_{i}",
                                                  size=(size[0], size[2]),
                                                  dtype=torch.float32,
                                                  keep_dimensions=True)
                        self._rnn_tensors_names.append(f"rnn_value_{i}")
                    self._rnn_initial_states["value"].append(
                        torch.zeros(size, dtype=torch.float32, device=self.device))

        self._current_log_prob    = None
        self._current_next_states = None

    # -------------------------------------------------------------------------
    # act
    # -------------------------------------------------------------------------

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}

        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states), **rnn}, role="policy")

        actions, log_prob, outputs = self.policy.act(
            {"states": self._state_preprocessor(states), **rnn}, role="policy")
        self._current_log_prob = log_prob

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        return actions, log_prob, outputs

    # -------------------------------------------------------------------------
    # record_transition
    # -------------------------------------------------------------------------

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        super().record_transition(states, actions, rewards, next_states,
                                  terminated, truncated, infos, timestep, timesteps)

        # Firing rate logging.
        if hasattr(self.policy, "_last_firing_rate"):
            self.track_data("SNN/firing_rate_mean", self.policy._last_firing_rate)

        # Entropy decay step.
        self._entropy_loss_scale = self._entropy_scheduler.get(timestep)

        
        if hasattr(infos, "get") and infos.get("log"):
            for key, val in infos["log"].items():
                if isinstance(val, (int, float)):
                    self.track_data(key, val)
                elif hasattr(val, "item"):
                    self.track_data(key, val.item())

        self._log_counter += 1

        if self.memory is not None:
            self._current_next_states = next_states

            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            values, _, outputs = self.value.act(
                {"states": self._state_preprocessor(states), **rnn}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            rnn_states = {}
            if self._rnn:
                rnn_states.update({
                    f"rnn_policy_{i}": s.transpose(0, 1)
                    for i, s in enumerate(self._rnn_initial_states["policy"])
                })
                if self.policy is not self.value:
                    rnn_states.update({
                        f"rnn_value_{i}": s.transpose(0, 1)
                        for i, s in enumerate(self._rnn_initial_states["value"])
                    })

            self.memory.add_samples(
                states=states, actions=actions, rewards=rewards,
                next_states=next_states, terminated=terminated, truncated=truncated,
                log_prob=self._current_log_prob, values=values, **rnn_states)

            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states, actions=actions, rewards=rewards,
                    next_states=next_states, terminated=terminated, truncated=truncated,
                    log_prob=self._current_log_prob, values=values, **rnn_states)

        # Advance RNN states and zero out the ones whose episode just ended.
        if self._rnn:
            if self.policy is self.value:
                self._rnn_final_states["value"] = self._rnn_final_states["policy"]
            else:
                self._rnn_final_states["value"] = outputs.get("rnn", [])

            # Reset the membrane potential for envs whose episode terminated.
            finished = terminated.nonzero(as_tuple=False)
            if finished.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

    # -------------------------------------------------------------------------
    # pre/post interaction
    # -------------------------------------------------------------------------

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")
        super().post_interaction(timestep, timesteps)

    # -------------------------------------------------------------------------
    # _update
    # -------------------------------------------------------------------------

    def _update(self, timestep: int, timesteps: int) -> None:

        def compute_gae(rewards, dones, values, next_values,
                        discount_factor=0.99, lambda_coefficient=0.95):
            advantage  = 0
            advantages = torch.zeros_like(rewards)
            not_dones  = dones.logical_not()
            memory_size = rewards.shape[0]
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage   = (rewards[i] - values[i]
                               + discount_factor * not_dones[i]
                               * (next_values + lambda_coefficient * advantage))
                advantages[i] = advantage
            returns    = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        # Compute the bootstrap value at the end of the rollout.
        with torch.no_grad():
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float()), **rnn},
                role="value")
            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values",     self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns",    self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # Sample mini-batches
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names,
            mini_batches=self._mini_batches,
            sequence_length=self._rnn_sequence_length,
        )

        rnn_policy, rnn_value = {}, {}
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self._mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss  = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss   = 0

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for i, (sampled_states, sampled_actions, sampled_dones,
                    sampled_log_prob, sampled_values,
                    sampled_returns, sampled_advantages) in enumerate(sampled_batches):

                if self._rnn:
                    if self.policy is self.value:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                                      "terminated": sampled_dones}
                        rnn_value  = rnn_policy
                    else:
                        rnn_policy = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n],
                                      "terminated": sampled_dones}
                        rnn_value  = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "value" in n],
                                      "terminated": sampled_dones}

                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                _, next_log_prob, _ = self.policy.act(
                    {"states": sampled_states, "taken_actions": sampled_actions, **rnn_policy},
                    role="policy")

                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # Entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # Policy loss
                ratio              = torch.exp(next_log_prob - sampled_log_prob)
                surrogate          = sampled_advantages * ratio
                surrogate_clipped  = sampled_advantages * torch.clip(
                    ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                policy_loss        = -torch.min(surrogate, surrogate_clipped).mean()

                # Value loss
                predicted_values, _, _ = self.value.act(
                    {"states": sampled_states, **rnn_value}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(
                        predicted_values - sampled_values,
                        min=-self._value_clip, max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # Optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self._grad_norm_clip)
                self.optimizer.step()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss  += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveRL):
                    self.scheduler.step(torch.tensor(kl_divergences).mean())
                else:
                    self.scheduler.step()

        # Logging
        self.track_data("Loss / Policy loss",
                        cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss",
                        cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss",
                            cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Policy / Standard deviation",
                        self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])