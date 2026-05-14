"""
Shared policy/value SNN backbone for the Crazyflie PPO agent.

SNNModel stacks configurable LIF hidden layers between a spike encoder and a
NoSpikingLIF output layer. All membrane potentials are exposed as explicit
recurrent state so the agent can persist them across timesteps and reset them
at episode termination.
"""

from omegaconf import DictConfig
from typing import Tuple, Optional
import torch
import torch.nn as nn
import snntorch as snn
from src.model.surrogate_gradient import rectangular_sg
from src.model.rl_mlp import MLPPolicy, MLPValue
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from src.model.encoding import build_encoder


# =============================================================================
# Utilities
# =============================================================================

def _prep_mem(
    mem_raw: Optional[list],
    num_layers: int,
    num_envs: int,
    hidden: int,
    num_actions: int,
    device: torch.device,
) -> list:
    """
    Memory slots: [encoder, hidden_0..N, out_policy, out_value].
    Length: num_layers + 3.
    """
    if mem_raw is None or len(mem_raw) == 0:
        return (
            [torch.zeros(num_envs, hidden, device=device)
             for _ in range(num_layers + 1)]
            + [torch.zeros(num_envs, num_actions, device=device)]
            + [torch.zeros(num_envs, 1, device=device)]
        )
    return [m.squeeze(0) if m.dim() == 3 else m for m in mem_raw]


def _reshape_for_update(states: torch.Tensor, N: int) -> torch.Tensor:
    L = states.shape[0] // N
    return states.view(N, L, -1).transpose(0, 1)


# =============================================================================
# Non-spiking LIF (vectorized)
# =============================================================================

class NoSpikingLIF(nn.Module):

    def __init__(self, in_size: int, out_size: int, beta: float, learn_beta: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        beta_init = torch.full((out_size,), beta)
        if learn_beta:
            self.beta = nn.Parameter(beta_init)
        else:
            self.register_buffer("beta", beta_init)

    def forward(self, spk: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rollout path: single time step, 2D input.
        if spk.dim() == 2:
            new_mem = self.beta * mem + self.fc(spk)
            return new_mem, new_mem

        # PPO update path: time-major 3D sequence.
        L, N, _ = spk.shape

        # Apply the linear layer to the whole sequence at once.
        current_seq = self.fc(spk)

        mem_steps = []
        curr_mem = mem

        # Only the membrane accumulation stays in the loop.
        for t in range(L):
            curr_mem = self.beta * curr_mem + current_seq[t]
            mem_steps.append(curr_mem)

        out_seq = torch.stack(mem_steps)
        return out_seq, curr_mem


# =============================================================================
# SNN backbone
# =============================================================================

class SNNBackbone(nn.Module):

    def __init__(self, cfg: DictConfig, num_observations: int, device: torch.device):
        super().__init__()

        self.hidden     = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.device     = device

        spike_grad_rect = rectangular_sg(a=0.5)
        self.encoder    = build_encoder(cfg, num_observations, spike_grad_rect)

        self.fc_hidden = nn.ModuleList([
            nn.Linear(self.hidden, self.hidden)
            for _ in range(self.num_layers)
        ])

        self.lif_layers = nn.ModuleList([
            snn.Leaky(
                beta=cfg.beta,
                threshold=cfg.threshold,
                reset_mechanism=cfg.reset_mechanism,
                learn_beta=cfg.learn_beta,
                learn_threshold=cfg.learn_threshold,
                spike_grad=spike_grad_rect,
            )
            for _ in range(self.num_layers)
        ])

        self.last_firing_rate = 0.0

    def forward(self, x: torch.Tensor, mem: list) -> Tuple[torch.Tensor, list]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        L, N, _ = x.shape
        outputs  = []
        cur_mem  = list(mem)

        for t in range(L):
            spk, m0 = self.encoder(x[t], cur_mem[0])
            next_mem = [m0]
            for i in range(self.num_layers):
                spk, m_i = self.lif_layers[i](self.fc_hidden[i](spk), cur_mem[i + 1])
                next_mem.append(m_i)
            outputs.append(spk)
            cur_mem = next_mem

        self.last_firing_rate = spk.mean().item()
        features = torch.stack(outputs)
        if L == 1:
            features = features.squeeze(0)

        return features, cur_mem


# =============================================================================
# SNN model: policy and value share one instance, with a built-in cache.
# =============================================================================

class SNNModel(GaussianMixin, DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, cfg, num_envs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
                               min_log_std=-20.0, max_log_std=0.0)
        DeterministicMixin.__init__(self)

        self.num_envs        = num_envs
        self.sequence_length = cfg.sequence_length

        self.backbone = SNNBackbone(cfg, self.num_observations, device)

        self.out_lif_policy = NoSpikingLIF(cfg.hidden_size, self.num_actions,
                                           cfg.beta, cfg.learn_beta)
        self.out_lif_value  = NoSpikingLIF(cfg.hidden_size, 1,
                                           cfg.beta, cfg.learn_beta)

        self.log_std_parameter = nn.Parameter(-1 * torch.ones(self.num_actions))
        self._last_firing_rate = 0.0

        # Cache fields: compute() is called twice per step (policy + value),
        # both with the same inputs; this lets the second call reuse the result.
        self._cache_states_ptr = None
        self._cache_mem_id = None
        self._cache_result = None

    def reset_hidden_states(self, env_ids):
        pass

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        value, _, outputs = self.compute(inputs, role)
        return value, None, outputs

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length, "sizes":
            [(1, self.num_envs, self.backbone.hidden)] * (self.backbone.num_layers + 1)
            + [(1, self.num_envs, self.num_actions)]
            + [(1, self.num_envs, 1)]
        }}

    def compute(self, inputs, role):
        states = inputs["states"]
        mem_raw = inputs.get("rnn")
        mem    = _prep_mem(mem_raw, self.backbone.num_layers,
                           self.num_envs, self.backbone.hidden,
                           self.num_actions, self.device)

        # Cache lookup. id(mem[0]) is unreliable when tensors are reused; the
        # underlying data pointer is the only stable identity here.
        current_ptr = states.data_ptr()
        current_mem_id = mem[0].data_ptr()

        if self._cache_states_ptr == current_ptr and self._cache_mem_id == current_mem_id:
            mean_actions, value, stored_mem = self._cache_result

        else:
            # Cache miss: run the full forward pass.
            out_mem_policy = mem[-2]
            out_mem_value  = mem[-1]
            bb_mem         = mem[:-2]

            batch, N = states.shape[0], bb_mem[0].shape[0]

            if batch != N:
                # PPO update path.
                states = _reshape_for_update(states, N)
                spk_seq, new_bb_mem = self.backbone(states, bb_mem)

                # Vectorized output heads.
                policy_seq, out_mem_policy = self.out_lif_policy(spk_seq, out_mem_policy)
                value_seq, out_mem_value   = self.out_lif_value(spk_seq, out_mem_value)

                mean_actions = policy_seq.transpose(0, 1).reshape(batch, self.num_actions)
                value        = value_seq.transpose(0, 1).reshape(batch, 1)
            else:
                # Rollout path.
                spk, new_bb_mem = self.backbone(states, bb_mem)

                mean_actions, out_mem_policy = self.out_lif_policy(spk, out_mem_policy)
                value, out_mem_value         = self.out_lif_value(spk, out_mem_value)

            self._last_firing_rate = self.backbone.last_firing_rate

            stored_mem = (
                [m.unsqueeze(0) for m in new_bb_mem]
                + [out_mem_policy.unsqueeze(0)]
                + [out_mem_value.unsqueeze(0)]
            )

            # Refresh the cache for the second compute() call of the same step.
            self._cache_states_ptr = current_ptr
            self._cache_mem_id = current_mem_id
            self._cache_result = (mean_actions, value, stored_mem)

        # Return policy or value head depending on the requested role.
        if role == "policy":
            return mean_actions, self.log_std_parameter, {"rnn": stored_mem}
        else:
            return value, self.log_std_parameter, {"rnn": stored_mem}


# =============================================================================
# Factory
# =============================================================================

def build_model(obs_space, act_space, device, cfg, num_envs):
    if cfg.type == "snn":
        model  = SNNModel(obs_space, act_space, device, cfg, num_envs)

        policy = model
        value  = model
    elif cfg.type == "mlp":
        policy = MLPPolicy(obs_space, act_space, device, cfg)
        value  = MLPValue(obs_space, act_space, device, cfg)
    else:
        raise ValueError(f"unknown net.type: '{cfg.type}'")
    return policy, value
