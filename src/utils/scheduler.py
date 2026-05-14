import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmStartCosineRestarts(_LRScheduler):
    """
    Long initial cosine cycle (T_warm) followed by warm restarts every
    T_restart steps.

    Example with T_warm=150k, T_restart=50k, timesteps=400k:
      [0     -> 150k]: cosine from LR_max down to eta_min
      [150k  -> 200k]: cosine from LR_max down to eta_min
      [200k  -> 250k]: cosine from LR_max down to eta_min
      ...

    Args:
        T_warm:    length of the initial long cycle.
        T_restart: length of each subsequent cycle.
        eta_min:   minimum learning rate at the end of every cycle.
    """

    def __init__(self, optimizer, T_warm: int, T_restart: int, eta_min: float = 1e-5, last_epoch: int = -1):
        self.T_warm    = T_warm
        self.T_restart = T_restart
        self.eta_min   = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch

        if t <= self.T_warm:
            progress = t / self.T_warm
        else:
            progress = ((t - self.T_warm) % self.T_restart) / self.T_restart

        return [
            self.eta_min + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]