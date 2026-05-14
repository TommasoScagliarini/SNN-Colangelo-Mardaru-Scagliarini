import sys
import os
import glob

ORIGINAL_ARGV = sys.argv[:]
sys.argv = [
    arg for arg in sys.argv
    if arg.startswith("--") or "=" not in arg
]

from isaaclab.app import AppLauncher

APP_LAUNCHER_KIT_ARGS = (
    "--/app/vulkan=false "
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1"
)

launcher = AppLauncher(headless=False, kit_args=APP_LAUNCHER_KIT_ARGS)
simulation_app = launcher.app

sys.argv = ORIGINAL_ARGV

import hydra
from omegaconf import DictConfig, OmegaConf

from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from src.model.snn_model import build_model
from src.agent.ppo_snn import PPO_SNN
from src.utils.params import merge_params

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def find_latest_checkpoint(logs_dir: str = "logs") -> str:
    """
    Return the path to the most recent best_agent.pt under logs/.

    Searches logs/**/checkpoints/best_agent.pt and picks the file with the
    latest modification time.
    """
    pattern = os.path.join(PROJECT_ROOT, logs_dir, "**", "checkpoints", "best_agent.pt")
    checkpoints = glob.glob(pattern, recursive=True)

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {logs_dir}/. "
            f"Run training first: python main_hydra.py"
        )

    latest = max(checkpoints, key=os.path.getmtime)
    print(f"[play] checkpoint found: {latest}")
    return latest

@hydra.main(
    config_path="src/task/crazyflie",
    config_name="cfg",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = 16
    env_cfg.sim.device = "cuda:0"

    env = QuadcopterEnv(cfg=env_cfg, render_mode="human")
    env = wrap_env(env)

    agent_cfg = merge_params(
        params=OmegaConf.to_container(cfg.algorithm.agent, resolve=True),
        default_params={},
        env=env,
    )

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

    trainer_cfg = {"timesteps": 3000, "headless": False}
    agent_cfg["timesteps"] = trainer_cfg["timesteps"]

    agent = PPO_SNN(
        models={"policy": policy, "value": value},
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        cfg=agent_cfg,
    )

    if cfg.agent_path:
        checkpoint = os.path.join(PROJECT_ROOT, cfg.agent_path)
    else:
        checkpoint = find_latest_checkpoint()

    agent.load(checkpoint)

    print(f"[play] checkpoint loaded: {checkpoint}")

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={"timesteps": 3000, "headless": False},
    )
    trainer.eval()

    simulation_app.close()


if __name__ == "__main__":
    main()
