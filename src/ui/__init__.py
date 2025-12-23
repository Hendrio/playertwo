"""
UI package for Mario RL training dashboard.

This package provides:
- Backend: TrainingState, AudioManager (state management)
- Frontend: Components (GameView, Metrics, Controls, Chart)
- Pages: TrainingDashboard (main page layout)
"""

# Backend exports
from .state import TrainingState, get_training_state, reset_training_state
from .audio import AudioManager, AudioEvent, get_audio_manager

# Frontend component exports
from .components import GameViewComponent, MetricsPanel, ControlButtons, RewardsChart

# Page exports
from .pages import TrainingDashboard, create_dashboard, run_dashboard

__all__ = [
    # Backend
    "TrainingState",
    "get_training_state", 
    "reset_training_state",
    "AudioManager",
    "AudioEvent",
    "get_audio_manager",
    # Components
    "GameViewComponent",
    "MetricsPanel",
    "ControlButtons",
    "RewardsChart",
    # Pages
    "TrainingDashboard",
    "create_dashboard",
    "run_dashboard",
]
