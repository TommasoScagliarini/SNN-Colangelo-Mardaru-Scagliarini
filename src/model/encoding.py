"""
Spike encoding: convert continuous (floating-point) observations into spikes
suitable for the SNN.

Three strategies are provided:
  - DirectEncoder:  no temporal loop; the linear projection is fed directly to
                    the first LIF layer as input current (T = 1).
  - RateEncoder:    spike frequency is proportional to the input magnitude.
  - LatencyEncoder: the first spike arrives sooner for larger input values.

Used inside SNNBackbone:
    self.encoder = build_encoder(cfg)          # in __init__
    spikes = self.encoder(obs, self.mem[0])    # in forward()
"""

import torch
import torch.nn as nn
import snntorch as snn
from src.model.surrogate_gradient import rectangular_sg
from omegaconf import DictConfig

class DirectEncoder(nn.Module):
    """
    No dedicated encoding layer or temporal loop: the continuous observation is
    multiplied by a learnable linear layer and treated as the input current of
    the first LIF layer, with ticks = 1.
    """
    def __init__(
        self,
        num_observations: int,
        hidden_size: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
        learn_beta: bool,
        learn_threshold: bool,
        spike_grad: callable,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # This linear transform acts as the "first layer"; its weights are
        # updated by gradient descent to minimize the loss.
        self.fc = nn.Linear(num_observations, hidden_size)

        # LIF neuron that turns the current into a spike at tick T = 1.
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            spike_grad=spike_grad
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        # The continuous observation is projected to obtain the input current.
        cur = self.fc(x)

        # One single time step (tick = 1).
        spk, mem = self.lif(cur, mem)

        return spk, mem


# =============================================================================
# Rate encoder, the firing rate over T timesteps is proportional to the input magnitude:
#   1. fc projects obs into the hidden space.
#   2. The LIF neuron is driven by this current for T timesteps.
#   3. The higher the current, the more frequent the spikes.
# =============================================================================

class RateEncoder(nn.Module):
    """
    Convert continuous observations into a firing rate over T timesteps.

    Args:
        num_observations: input size (12 for Crazyflie).
        hidden_size:      output size (matches the backbone hidden size).
        T:                number of timesteps used for encoding.
        beta:             LIF membrane decay.
        threshold:        spike threshold.
        reset_mechanism:  "subtract" or "zero".
        learn_beta:       if True, beta is a learnable parameter.
        learn_threshold:  if True, threshold is a learnable parameter.
    """

    def __init__(
        self,
        num_observations: int,
        hidden_size: int,
        T: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
        learn_beta: bool,
        learn_threshold: bool,
    ):
        super().__init__()
        self.T = T
        self.hidden_size = hidden_size

        # Linear projection: obs -> input current for the LIF neuron.
        self.fc = nn.Linear(num_observations, hidden_size)

        # Encoding LIF neuron, kept separate from the backbone LIFs.
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        """
        Args:
            x:   (num_envs, num_observations)
            mem: (num_envs, hidden_size)
        Returns:
            firing rate clamped to [0.05, 1.0], shape (num_envs, hidden_size),
            together with the updated membrane state.
        """
        spk_sum = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        cur = self.fc(x)   # linear projection, sign-preserving

        for _ in range(self.T):
            spk, mem = self.lif(cur, mem)
            spk_sum += spk

        return (spk_sum / self.T).clamp(min=0.05), mem


# =============================================================================
# Latency encoder, the timing of the first spike encodes the input value:
#   1. Normalize obs into [0, 1].
#   2. The neuron receives a current that decays over time.
#   3. A higher input crosses the threshold sooner -> lower latency.
#   4. The timing of the first spike encodes the input value.
# =============================================================================

class LatencyEncoder(nn.Module):
    """
    Convert continuous observations into spikes via temporal coding.

    The current at timestep t is:  I(t) = x_norm * exp(-t / tau),
    where tau controls how fast the drive decays over time.
    """

    def __init__(
        self,
        num_observations: int,
        hidden_size: int,
        T: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
        learn_beta: bool,
        learn_threshold: bool,
        tau: float = 5.0,
    ):
        super().__init__()
        self.T   = T
        self.tau = tau
        self.hidden_size = hidden_size

        self.fc = nn.Linear(num_observations, hidden_size)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )

        # Precomputed time weights [exp(0), exp(-1/tau), ..., exp(-(T-1)/tau)].
        # Shape (T,); multiplied into the current at each timestep.
        t = torch.arange(T, dtype=torch.float32)
        self.register_buffer("time_weights", torch.exp(-t / tau))

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        x_norm  = torch.sigmoid(self.fc(x))
        spk_sum = torch.zeros(x.shape[0], self.hidden_size, device=x.device)

        for t in range(self.T):
            cur = x_norm * self.time_weights[t]
            spk, mem = self.lif(cur, mem)
            spk_sum += spk

        return (spk_sum / self.T).clamp(min=0.05), mem


# =============================================================================
# Factory, invoked from SNNBackbone.__init__().
# =============================================================================

def build_encoder(cfg: DictConfig, num_observations: int, spike_grad=None) -> nn.Module:
    common = dict(
        num_observations = num_observations,
        hidden_size      = cfg.hidden_size,
        beta             = cfg.beta,
        threshold        = cfg.threshold,
        reset_mechanism  = cfg.reset_mechanism,
        learn_beta       = cfg.learn_beta,
        learn_threshold  = cfg.learn_threshold,
    )

    if cfg.encoding == "direct":
        return DirectEncoder(**common, spike_grad=spike_grad)

    elif cfg.encoding == "rate":
        return RateEncoder(**common, T=cfg.encoding_timesteps)

    elif cfg.encoding == "latency":
        return LatencyEncoder(**common, T=cfg.encoding_timesteps, tau=5.0)

    else:
        raise ValueError(
            f"unknown encoding: '{cfg.encoding}'. Use 'direct', 'rate' or 'latency'."
        )
