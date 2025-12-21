"""
Learning rate and exploration schedules.
"""

from typing import Callable


class LinearSchedule:
    """
    Linear interpolation between initial and final values.
    """
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int
    ):
        """
        Initialize the linear schedule.
        
        Args:
            initial_value: Starting value
            final_value: Ending value
            total_steps: Number of steps for the transition
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
    
    def value(self, step: int) -> float:
        """
        Get the value at a given step.
        
        Args:
            step: Current step
            
        Returns:
            Interpolated value
        """
        if step >= self.total_steps:
            return self.final_value
        
        fraction = step / self.total_steps
        return self.initial_value + fraction * (self.final_value - self.initial_value)
    
    def __call__(self, step: int) -> float:
        """Alias for value()."""
        return self.value(step)


class ExponentialSchedule:
    """
    Exponential decay from initial to final value.
    """
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        decay_rate: float
    ):
        """
        Initialize exponential schedule.
        
        Args:
            initial_value: Starting value
            final_value: Minimum value
            decay_rate: Decay rate per step (0 < decay_rate < 1)
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_rate = decay_rate
    
    def value(self, step: int) -> float:
        """Get value at given step."""
        decayed = self.initial_value * (self.decay_rate ** step)
        return max(decayed, self.final_value)
    
    def __call__(self, step: int) -> float:
        return self.value(step)


def get_linear_lr_scheduler(
    optimizer,
    initial_lr: float,
    total_steps: int
) -> Callable[[int], None]:
    """
    Create a linear learning rate scheduler function.
    
    Args:
        optimizer: PyTorch optimizer
        initial_lr: Initial learning rate
        total_steps: Total training steps
        
    Returns:
        Function that updates optimizer LR given current step
    """
    schedule = LinearSchedule(initial_lr, 0.0, total_steps)
    
    def update_lr(step: int):
        new_lr = schedule(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    return update_lr


if __name__ == "__main__":
    # Test linear schedule
    linear = LinearSchedule(1.0, 0.1, 1000)
    print("Linear Schedule:")
    for step in [0, 250, 500, 750, 1000, 1500]:
        print(f"  Step {step}: {linear(step):.4f}")
    
    # Test exponential schedule
    exp = ExponentialSchedule(1.0, 0.1, 0.999)
    print("\nExponential Schedule:")
    for step in [0, 1000, 2000, 3000]:
        print(f"  Step {step}: {exp(step):.4f}")
