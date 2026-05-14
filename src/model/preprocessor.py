import torch
import math
from skrl.resources.preprocessors.torch import RunningStandardScaler


class CrazyflieSNNPreprocessor(RunningStandardScaler):
    """
    Static Min-Max normalizer that maps the 12-dimensional Crazyflie observation
    to [0, 1], as required by LIF neurons (negative inputs cause hyperpolarization).

    Subclasses RunningStandardScaler for skrl checkpoint compatibility.
    Must be used as state_preprocessor only: the value preprocessor receives
    scalars (size=1), incompatible with the 12-dim min/max tensors here.

    Observation layout (12 dims):
      pos_error (3) | quat_xyz (3) | linear_vel (3) | angular_vel (3)
    The quaternion w component is dropped (redundant for a unit quaternion).
    """

    def __init__(self, size, device, **kwargs):
        super().__init__(size=size, device=device, **kwargs)

        PI = math.pi
        self._min_vals = torch.tensor(
            [-5., -5., 0., -PI, -PI, -PI, -10., -10., -10., -20., -20., -20.],
            device=device
        )
        self._max_vals = torch.tensor(
            [5., 5., 5., PI, PI, PI, 10., 10., 10., 20., 20., 20.],
            device=device
        )
        self._range = self._max_vals - self._min_vals

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Normalize x into [0, 1] using the static Min-Max range."""
        norm = (x - self._min_vals) / (self._range + 1e-8)
        return torch.clamp(norm, 0.0, 1.0)
