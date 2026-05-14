from typing import Tuple
from omegaconf import DictConfig

import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

# =============================================================================
# MLP baseline
# =============================================================================

class MLPPolicy(GaussianMixin, Model):

    def __init__(self, observation_space, action_space, device, cfg: DictConfig):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)

        layers = [nn.Linear(self.num_observations, cfg.hidden_size), nn.ELU()]
        for _ in range(cfg.num_layers - 1):
            layers += [nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ELU()]
        layers += [nn.Linear(cfg.hidden_size, self.num_actions)]

        self.net               = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def reset_hidden_states(self, env_ids: torch.Tensor) -> None:
        pass

    def compute(self, inputs: dict, role: str) -> Tuple:
        return self.net(inputs["states"]), self.log_std_parameter, {}


class MLPValue(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space, device, cfg: DictConfig):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        layers = [nn.Linear(self.num_observations, cfg.hidden_size), nn.ELU()]
        for _ in range(cfg.num_layers - 1):
            layers += [nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ELU()]
        layers += [nn.Linear(cfg.hidden_size, 1)]

        self.net = nn.Sequential(*layers)

    def reset_hidden_states(self, env_ids: torch.Tensor) -> None:
        pass

    def compute(self, inputs: dict, role: str) -> Tuple:
        return self.net(inputs["states"]), {}
