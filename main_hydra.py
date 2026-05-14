import random
import numpy as np
import torch
import hydra
import sys
from omegaconf import DictConfig, OmegaConf
import time

from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import wrap_env

from src.model.snn_model import build_model
from src.agent.ppo_snn import PPO_SNN
from src.utils import set_global_path
from src.utils.params import merge_params, save_params


@hydra.main(
    config_path="src/task/crazyflie",
    config_name="cfg",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    # If cfg.seed is null, draw a random seed so the run is still reproducible.
    seed = cfg.seed if cfg.seed is not None else random.randint(0, 100_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[main] seed: {seed}")

    # Must be called before save_params() to ensure the output directory exists.
    set_global_path(
        root_folder=cfg.root_path if cfg.root_path else "logs",
        path_prefix=cfg.prefix if cfg.prefix else "crazyflie_snn_",
    )

    # IsaacLab's AppLauncher parses sys.argv with argparse and rejects Hydra's
    # key=value overrides; strip them before the env is loaded.
    sys.argv = [
        arg for arg in sys.argv
        if arg.startswith("--") or "=" not in arg
    ]

    env = load_isaaclab_env(
        task_name=cfg.task.name,
        num_envs=cfg.task.num_envs,
        headless=cfg.task.headless,
    )
    env = wrap_env(env)
    env.seed(seed)

    agent_cfg = merge_params(
        params=OmegaConf.to_container(cfg.algorithm.agent, resolve=True),
        default_params={},
        env=env,
    )

    agent_cfg["net_activity"]        = cfg.net_activity
    agent_cfg["net_actions"]         = cfg.net_actions
    agent_cfg["net_decay_threshold"] = cfg.net_decay_threshold

    if cfg.wandb is not None:
        agent_cfg["experiment"]["wandb"] = True
        agent_cfg["experiment"]["wandb_kwargs"] = {
            "project": cfg.wandb,
            "entity":  None,
        }

    save_params({
        "crazyflie": OmegaConf.to_container(cfg.task,      resolve=True),
        "net":       OmegaConf.to_container(cfg.net,       resolve=True),
        "ppo":       OmegaConf.to_container(cfg.algorithm, resolve=True),
        "cfg":       {"seed": seed, "wandb": cfg.wandb},
    })

    policy, value = build_model(
        obs_space=env.observation_space,
        act_space=env.action_space,
        device=env.device,
        cfg=cfg.net,
        num_envs=env.num_envs,
    )

    memory = RandomMemory(
        memory_size=agent_cfg["rollouts"],
        num_envs=env.num_envs,
        device=env.device,
    )

    trainer_cfg = OmegaConf.to_container(cfg.algorithm.trainer, resolve=True)
    agent_cfg["timesteps"] = trainer_cfg["timesteps"]  # needed by the entropy scheduler

    agent = PPO_SNN(
        models={
            "policy": policy,
            "value":  value,
        },
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg,
        num_envs=env.num_envs,
    )

    if cfg.agent_path is not None:
        agent.load(cfg.agent_path)
        print(f"[main] checkpoint loaded from: {cfg.agent_path}")

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()

