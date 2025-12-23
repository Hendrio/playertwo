"""
Custom reward wrapper for Super Mario Bros using Gymnasium API.

Implements the reward function:
    Reward = velocity + clock + death
    
Where:
    - velocity (v): x_t - x_{t-1} (encourage moving right)
    - clock (c): small negative penalty per frame
    - death (d): -15 penalty for dying
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper that implements:
    - Velocity reward: encourages moving right
    - Clock penalty: discourages wasting time
    - Death penalty: large penalty for dying
    """
    
    def __init__(
        self,
        env: gym.Env,
        velocity_scale: float = 1.0,
        clock_penalty: float = -0.01,
        death_penalty: float = -15.0
    ):
        """
        Initialize the reward wrapper.
        
        Args:
            env: The environment to wrap
            velocity_scale: Scale factor for velocity reward
            clock_penalty: Penalty per frame (should be negative)
            death_penalty: Penalty for dying (should be negative)
        """
        super().__init__(env)
        
        self.velocity_scale = velocity_scale
        self.clock_penalty = clock_penalty
        self.death_penalty = death_penalty
        
        # Track previous x position
        self._prev_x_pos = 0
        self._prev_life = 2  # Mario starts with 2 lives
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and initialize tracking variables."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset tracking
        self._prev_x_pos = 0
        self._prev_life = 2
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and compute custom reward.
        
        The original reward from gym-super-mario-bros is replaced with
        our custom reward function.
        """
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Get current state from info
        x_pos = info.get('x_pos', 0)
        life = info.get('life', 2)
        flag_get = info.get('flag_get', False)
        
        # Calculate velocity reward
        velocity = (x_pos - self._prev_x_pos) * self.velocity_scale
        
        # Clock penalty (per frame)
        clock = self.clock_penalty
        
        # Death penalty
        death = 0.0
        if life < self._prev_life:
            death = self.death_penalty
        
        # Compute total reward
        reward = velocity + clock + death
        
        # Bonus for completing the level
        if flag_get:
            reward += 100.0
        
        # Update tracking variables
        self._prev_x_pos = x_pos
        self._prev_life = life
        
        # Add reward components to info for logging
        info['reward_velocity'] = velocity
        info['reward_clock'] = clock
        info['reward_death'] = death
        info['reward_total'] = reward
        
        return obs, reward, terminated, truncated, info


def wrap_with_custom_reward(
    env: gym.Env,
    velocity_scale: float = 1.0,
    clock_penalty: float = -0.01,
    death_penalty: float = -15.0
) -> gym.Env:
    """
    Wrap an environment with custom reward function.
    
    Args:
        env: The environment to wrap
        velocity_scale: Scale factor for velocity reward
        clock_penalty: Penalty per frame
        death_penalty: Penalty for dying
        
    Returns:
        Wrapped environment
    """
    return CustomRewardWrapper(
        env,
        velocity_scale=velocity_scale,
        clock_penalty=clock_penalty,
        death_penalty=death_penalty
    )


if __name__ == "__main__":
    from env import make_mario_env
    
    # Create environment with custom rewards
    env = make_mario_env()
    env = wrap_with_custom_reward(env, velocity_scale=1.0, clock_penalty=-0.01, death_penalty=-15.0)
    
    obs, info = env.reset()
    
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}: reward={reward:.2f}, x_pos={info.get('x_pos', 0)}")
        
        if done:
            print(f"Episode done! Total reward: {total_reward:.2f}")
            break
    
    env.close()
