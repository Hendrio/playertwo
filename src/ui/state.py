"""
Shared state manager for communication between training loop and UI.

Provides thread-safe data structures for real-time updates.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_num: int
    total_reward: float
    length: int
    x_pos: int
    timestamp: float


@dataclass  
class StepMetrics:
    """Metrics for a single step."""
    step: int
    reward: float
    velocity_reward: float
    clock_penalty: float
    death_penalty: float
    action: int
    x_pos: int


class TrainingState:
    """
    Thread-safe shared state for training visualization.
    
    Manages communication between the training thread and the UI.
    """
    
    def __init__(self, max_frames: int = 10, max_history: int = 1000):
        """
        Initialize training state.
        
        Args:
            max_frames: Maximum frames to keep in buffer
            max_history: Maximum history length for metrics
        """
        self._lock = threading.Lock()
        
        # Frame buffer (for game visualization)
        self._frame_buffer: deque = deque(maxlen=max_frames)
        self._current_frame: Optional[np.ndarray] = None
        
        # Step metrics history
        self._step_history: deque = deque(maxlen=max_history)
        
        # Episode metrics history
        self._episode_history: deque = deque(maxlen=100)
        
        # Current training info
        self._global_step: int = 0
        self._episode_count: int = 0
        self._current_episode_reward: float = 0.0
        self._current_episode_length: int = 0
        self._learning_rate: float = 0.0
        self._epsilon: float = 1.0
        self._policy_loss: float = 0.0
        self._value_loss: float = 0.0
        self._entropy: float = 0.0
        
        # Control flags
        self._is_training: bool = False
        self._should_stop: bool = False
        self._is_paused: bool = False
    
    # Frame management
    def push_frame(self, frame: np.ndarray):
        """Push a new frame to the buffer (RGB, HWC format)."""
        with self._lock:
            self._current_frame = frame.copy()
            self._frame_buffer.append(frame.copy())
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None
    
    # Step metrics
    def record_step(
        self,
        step: int,
        reward: float,
        velocity_reward: float = 0.0,
        clock_penalty: float = 0.0,
        death_penalty: float = 0.0,
        action: int = 0,
        x_pos: int = 0
    ):
        """Record metrics for a training step."""
        with self._lock:
            self._global_step = step
            self._current_episode_reward += reward
            self._current_episode_length += 1
            
            metrics = StepMetrics(
                step=step,
                reward=reward,
                velocity_reward=velocity_reward,
                clock_penalty=clock_penalty,
                death_penalty=death_penalty,
                action=action,
                x_pos=x_pos
            )
            self._step_history.append(metrics)
    
    def get_recent_steps(self, n: int = 100) -> List[StepMetrics]:
        """Get the most recent n step metrics."""
        with self._lock:
            return list(self._step_history)[-n:]
    
    # Episode metrics
    def record_episode_end(self, x_pos: int = 0):
        """Record end of episode."""
        with self._lock:
            self._episode_count += 1
            
            episode = EpisodeMetrics(
                episode_num=self._episode_count,
                total_reward=self._current_episode_reward,
                length=self._current_episode_length,
                x_pos=x_pos,
                timestamp=time.time()
            )
            self._episode_history.append(episode)
            
            # Reset current episode tracking
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
    
    def get_recent_episodes(self, n: int = 10) -> List[EpisodeMetrics]:
        """Get the most recent n episodes."""
        with self._lock:
            return list(self._episode_history)[-n:]
    
    # Training info updates
    def update_training_info(
        self,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None
    ):
        """Update training hyperparameters and losses."""
        with self._lock:
            if learning_rate is not None:
                self._learning_rate = learning_rate
            if epsilon is not None:
                self._epsilon = epsilon
            if policy_loss is not None:
                self._policy_loss = policy_loss
            if value_loss is not None:
                self._value_loss = value_loss
            if entropy is not None:
                self._entropy = entropy
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get current training info snapshot."""
        with self._lock:
            return {
                'global_step': self._global_step,
                'episode_count': self._episode_count,
                'current_episode_reward': self._current_episode_reward,
                'current_episode_length': self._current_episode_length,
                'learning_rate': self._learning_rate,
                'epsilon': self._epsilon,
                'policy_loss': self._policy_loss,
                'value_loss': self._value_loss,
                'entropy': self._entropy,
                'is_training': self._is_training,
                'is_paused': self._is_paused
            }
    
    # Control methods
    def start_training(self):
        """Signal that training has started."""
        with self._lock:
            self._is_training = True
            self._should_stop = False
    
    def stop_training(self):
        """Signal to stop training."""
        with self._lock:
            self._should_stop = True
    
    def pause_training(self):
        """Toggle pause state."""
        with self._lock:
            self._is_paused = not self._is_paused
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        with self._lock:
            return self._should_stop
    
    def is_paused(self) -> bool:
        """Check if training is paused."""
        with self._lock:
            return self._is_paused
    
    def is_training(self) -> bool:
        """Check if training is active."""
        with self._lock:
            return self._is_training
    
    def training_ended(self):
        """Signal that training has ended."""
        with self._lock:
            self._is_training = False


# Global state instance
_global_state: Optional[TrainingState] = None


def get_training_state() -> TrainingState:
    """Get or create the global training state."""
    global _global_state
    if _global_state is None:
        _global_state = TrainingState()
    return _global_state


def reset_training_state():
    """Reset the global training state."""
    global _global_state
    _global_state = TrainingState()
