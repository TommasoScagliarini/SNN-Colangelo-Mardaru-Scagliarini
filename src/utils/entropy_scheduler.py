class LinearEntropyDecay:
    """
    Linearly anneal the PPO entropy bonus coefficient from `start` to `end`
    over `total_steps` environment steps. A high initial value encourages
    exploration early in training; decaying it to zero lets the policy converge.
    Called at each step in ppo_snn.py via record_transition().

    Example: start=0.005, end=0.0, total_steps=400_000
      step 0       -> entropy_scale = 0.005
      step 200_000 -> entropy_scale = 0.0025
      step 400_000 -> entropy_scale = 0.0
    """

    def __init__(self, start: float, end: float, total_steps: int):
        self.start       = start
        self.end         = end
        self.total_steps = total_steps

    def get(self, current_step: int) -> float:
        progress = min(current_step / self.total_steps, 1.0)
        return self.start + (self.end - self.start) * progress