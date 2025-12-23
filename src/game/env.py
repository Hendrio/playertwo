"""
Super Mario Bros environment wrappers using Gymnasium API.

Provides preprocessing, frame stacking, and frame skipping functionality
for the gym-super-mario-bros environment via shimmy compatibility layer.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Tuple, Dict, Any

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import shimmy


# Custom action space: Right, Right+Jump, Right+Run, Jump
MARIO_ACTIONS = [
    ['right'],           # Right
    ['right', 'A'],      # Right + Jump
    ['right', 'B'],      # Right + Run
    ['A'],               # Jump only
]


class GymV21ToGymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to convert old gym v21 API to gymnasium API.
    
    gym-super-mario-bros uses the old API:
        - reset() returns obs
        - step() returns (obs, reward, done, info)
    
    gymnasium expects:
        - reset() returns (obs, info)
        - step() returns (obs, reward, terminated, truncated, info)
    """
    
    def __init__(self, env):
        # Don't call super().__init__ since env is not a gymnasium env
        self._env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=env.observation_space.shape, dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)
    
    @property
    def unwrapped(self):
        return self._env.unwrapped
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and return (obs, info) tuple."""
        obs = self._env.reset()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and return (obs, reward, terminated, truncated, info) tuple."""
        obs, reward, done, info = self._env.step(action)
        # In old API, done = terminated (no truncation concept)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()


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
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and fill buffer with initial observation."""
        obs, info = self.env.reset(**kwargs)
        
        # Fill buffer with initial frame
        for _ in range(self.n_frames):
            self.frames.append(obs)
        
        return self._get_observation(), info
    
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
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action for skip frames."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info


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
        Wrapped environment with gymnasium API
    """
    # Create base environment (old gym API)
    env = gym_super_mario_bros.make(env_name)
    
    # Apply custom action space
    env = JoypadSpace(env, MARIO_ACTIONS)
    
    # Convert to gymnasium API
    env = GymV21ToGymnasiumWrapper(env)
    
    # Apply frame skip
    env = FrameSkipWrapper(env, skip=frame_skip)
    
    # Apply grayscale and resize
    env = GrayscaleResizeWrapper(env, width=frame_size, height=frame_size)
    
    # Apply frame stacking
    env = FrameStackWrapper(env, n_frames=frame_stack)
    
    # Normalize observations
    env = NormalizeObservationWrapper(env)
    
    return env


if __name__ == "__main__":
    # Test the environment
    env = make_mario_env()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Step {i+1}: action={action}, reward={reward:.2f}, done={done}")
        
        if done:
            obs, info = env.reset()
    
    env.close()
