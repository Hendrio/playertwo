"""
Super Mario Bros environment wrappers.

Provides preprocessing, frame stacking, and frame skipping functionality
for the gym-super-mario-bros environment.
"""

import gym
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional, Dict, Any

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# Custom action space: Right, Right+Jump, Right+Run, Jump
MARIO_ACTIONS = [
    ['right'],           # Right
    ['right', 'A'],      # Right + Jump
    ['right', 'B'],      # Right + Run
    ['A'],               # Jump only
]


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale and resizes to target size.
    """
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width),
            dtype=np.uint8
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert to grayscale and resize."""
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Resize to target dimensions
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        return resized


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Stack n consecutive frames together.
    
    Output shape: (n, height, width)
    """
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        
        # Get the wrapped observation shape
        obs_shape = env.observation_space.shape
        
        # Create frame buffer
        self.frames = deque(maxlen=n_frames)
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(n_frames,) + obs_shape,
            dtype=np.uint8
        )
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset and fill buffer with initial observation."""
        obs = self.env.reset(**kwargs)
        
        # Fill buffer with initial frame
        for _ in range(self.n_frames):
            self.frames.append(obs)
        
        return self._get_observation()
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add new frame to buffer."""
        self.frames.append(obs)
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Stack frames from buffer."""
        return np.stack(list(self.frames), axis=0)


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeat action for n frames and accumulate rewards.
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action for skip frames."""
        total_reward = 0.0
        done = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return obs, total_reward, done, info


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Normalize observations to [0, 1] range and convert to float32.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Update observation space to float
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        return obs.astype(np.float32) / 255.0


class MarioEnvWrapper(gym.Wrapper):
    """
    Full Mario environment wrapper combining all preprocessing steps.
    
    Applies in order:
    1. Custom action space
    2. Frame skip
    3. Grayscale + resize
    4. Frame stacking
    5. Normalization
    """
    
    def __init__(
        self,
        env_name: str = "SuperMarioBros-1-1-v0",
        frame_skip: int = 4,
        frame_stack: int = 4,
        frame_size: int = 84
    ):
        # Create base environment
        env = gym_super_mario_bros.make(env_name)
        
        # Apply custom action space
        env = JoypadSpace(env, MARIO_ACTIONS)
        
        # Apply frame skip
        env = FrameSkipWrapper(env, skip=frame_skip)
        
        # Apply grayscale and resize
        env = GrayscaleResizeWrapper(env, width=frame_size, height=frame_size)
        
        # Apply frame stacking
        env = FrameStackWrapper(env, n_frames=frame_stack)
        
        # Normalize observations
        env = NormalizeObservationWrapper(env)
        
        super().__init__(env)
        
        self.env_name = env_name
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frame_size = frame_size


def make_mario_env(
    env_name: str = "SuperMarioBros-1-1-v0",
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_size: int = 84
) -> gym.Env:
    """
    Create a Mario environment with all preprocessing.
    
    Args:
        env_name: Name of the Mario environment
        frame_skip: Number of frames to skip
        frame_stack: Number of frames to stack
        frame_size: Size of resized frames
        
    Returns:
        Wrapped environment
    """
    return MarioEnvWrapper(
        env_name=env_name,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        frame_size=frame_size
    )


if __name__ == "__main__":
    # Test the environment
    env = make_mario_env()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, done={done}")
        
        if done:
            obs = env.reset()
    
    env.close()
