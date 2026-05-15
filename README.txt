================================================================================
SNN-Colangelo-Mardaru-Scagliarini
PPO Spiking Neural Network agent for Crazyflie quadcopter control in Isaac Lab.
================================================================================


--------------------------------------------------------------------------------
INSTALLATION REQUIREMENTS
--------------------------------------------------------------------------------

Hardware
  - NVIDIA GPU with CUDA support (recommended). CPU fallback supported via the
    Hydra overrides documented below.
  - At least 8 GB of RAM (16 GB or more recommended for the default num_envs).

Software
  - Python 3.10 or newer
  - NVIDIA Isaac Lab 5.1 (with Isaac Sim 5.1)
  - PyTorch (CUDA build coherent with the installed Isaac Lab version)
  - skrl       >= 1.4
  - snntorch   >= 0.9
  - hydra-core >= 1.3
  - omegaconf  >= 2.3
  - numpy

Refer to the official Isaac Lab documentation for the simulator installation:
https://isaac-sim.github.io/IsaacLab/


--------------------------------------------------------------------------------
REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

.
├── main_hydra.py              Training entrypoint (Hydra-driven).
├── play.py                    Evaluation entrypoint (loads a checkpoint).
├── commands.txt               Quick reference for training and testing commands.
├── config_base.txt            Baseline hyperparameter reference.
├── .gitignore
├── README.txt
└── src/
    ├── agent/
    │   └── ppo_snn.py         PPO agent for SNNs (adapted from skrl PPO_RNN).
    ├── model/
    │   ├── snn_model.py       Shared policy/value SNN backbone.
    │   ├── encoding.py        Spike encoders: direct, rate, latency.
    │   ├── preprocessor.py    Static Min-Max normalizer for LIF inputs.
    │   ├── surrogate_gradient.py   Rectangular surrogate gradient.
    │   └── rl_mlp.py          MLP baseline (non-SNN).
    ├── utils/
    │   ├── params.py          Converts YAML strings into skrl/torch objects.
    │   ├── scheduler.py       WarmStartCosineRestarts learning-rate scheduler.
    │   ├── entropy_scheduler.py    Linear decay of the PPO entropy bonus.
    │   └── global_variables.py     Run-level output directory management.
    └── task/
        └── crazyflie/
            ├── cfg.yaml              Top-level run configuration.
            ├── algorithm/ppo.yaml    PPO hyperparameters.
            ├── net/snn.yaml          SNN architecture.
            └── task/crazyflie.yaml   Isaac Lab task reference.

Runtime directories created on first run (excluded from git):
  logs/         TensorBoard events and checkpoints per training run.
  outputs/      Hydra-resolved configs and working directories.
  checkpoints/  Optional global checkpoint storage.


--------------------------------------------------------------------------------
DEMO INSTRUCTIONS
--------------------------------------------------------------------------------

Prerequisite: activate the conda environment that contains Isaac Lab and the
Python dependencies listed above.

1. Train an agent from scratch

       python main_hydra.py

   Default run: 200 000 timesteps, 4096 parallel environments. Checkpoints are
   written under  logs/crazyflie_snn_<timestamp>/checkpoints/best_agent.pt .

   Common Hydra overrides:

       python main_hydra.py task.num_envs=64                # fewer parallel envs
       python main_hydra.py task.headless=false             # render during training
       python main_hydra.py seed=123                        # custom seed
       python main_hydra.py algorithm.trainer.timesteps=400000
       python main_hydra.py net.type=mlp                    # MLP baseline
       python main_hydra.py sim_device=cpu rl_device=cpu    # CPU only

   Quick smoke run (a few seconds, useful to verify the installation):

       python main_hydra.py task.num_envs=4 task.headless=true \
                           algorithm.trainer.timesteps=64 \
                           algorithm.agent.rollouts=16

2. Evaluate the trained agent

   By default play.py loads the most recent best_agent.pt found under logs/:

       python play.py

   To load a specific checkpoint:

       python play.py agent_path=logs/crazyflie_snn_<run>/checkpoints/best_agent.pt

3. Visualize training curves

       tensorboard --logdir logs/crazyflie_snn

The complete command reference is also available in  commands.txt .


--------------------------------------------------------------------------------
CREDITS
--------------------------------------------------------------------------------

This project builds on the following open-source resources.

[1] Barchi F., Parisi E., Zanatta L., Bartolini A., Acquaviva A.
    "Energy efficient and low-latency spiking neural networks on embedded
     microcontrollers through spiking activity tuning."
    The Spiking Neural Network model and the rectangular surrogate gradient
    used in this work are inspired by the methodology described in this paper.

[2] NVIDIA Isaac Lab — robotics learning framework providing the quadcopter
    direct-control environment used here.
    https://github.com/isaac-sim/IsaacLab

[3] skrl — modular reinforcement learning library used as the base for the
    PPO implementation extended in src/agent/ppo_snn.py.
    https://github.com/Toni-SM/skrl

[4] snntorch — spiking neural network primitives for PyTorch.
    https://github.com/jeshraghian/snntorch

Authors: Ivan Colangelo, Razvan Stefan Mardaru, Tommaso Scagliarini.
