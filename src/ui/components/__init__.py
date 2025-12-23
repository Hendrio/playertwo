"""Components package for dashboard UI elements."""

from .game_view import GameViewComponent
from .metrics_panel import MetricsPanel
from .controls import ControlButtons
from .rewards_chart import RewardsChart

__all__ = [
    "GameViewComponent",
    "MetricsPanel", 
    "ControlButtons",
    "RewardsChart",
]
