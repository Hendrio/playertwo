"""
Training dashboard page - composed from modular components.

This is a simplified version of the original dashboard that uses
the modular components for better maintainability.
"""

import asyncio
import logging
from typing import Optional

from nicegui import ui, app

from ..state import TrainingState, get_training_state
from ..audio import AudioManager, AudioEvent, get_audio_manager
from ..styles import DASHBOARD_CSS
from ..components import GameViewComponent, MetricsPanel, ControlButtons, RewardsChart


logger = logging.getLogger("dashboard")


class TrainingDashboard:
    """
    Real-time training visualization dashboard.
    
    Composes modular components:
    - GameViewComponent: Live game frame display
    - MetricsPanel: Training statistics
    - ControlButtons: Start/Pause/Stop/Mute
    - RewardsChart: Rewards visualization
    """
    
    def __init__(self, state: Optional[TrainingState] = None):
        """
        Initialize the dashboard.
        
        Args:
            state: TrainingState instance (uses global if None)
        """
        logger.info("Initializing TrainingDashboard")
        self.state = state or get_training_state()
        self.audio = get_audio_manager()
        
        # Components
        self.game_view = GameViewComponent()
        self.metrics = MetricsPanel()
        self.controls = ControlButtons(
            on_start=self._on_start,
            on_pause=self._on_pause,
            on_stop=self._on_stop,
            on_toggle_mute=self._on_toggle_mute
        )
        self.chart = RewardsChart()
        
        # Update timer
        self.update_interval = 0.2  # 5 FPS
    
    def build(self) -> None:
        """Build the dashboard UI by composing components."""
        logger.info("Building dashboard UI")
        
        # Apply dark theme and styling
        ui.dark_mode().enable()
        ui.add_head_html(DASHBOARD_CSS)
        
        with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-4'):
            # Header
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('🎮 Mario RL Training Dashboard').classes('text-3xl font-bold text-white')
                self.controls.build()
            
            # Build audio elements (hidden)
            self.audio.build()
            
            # Main content row
            with ui.row().classes('w-full gap-4'):
                # Left column: Game view
                self.game_view.build()
                
                # Right column: Metrics
                self.metrics.build()
            
            # Rewards chart
            self.chart.build()
        
        # Start periodic updates
        ui.timer(self.update_interval, self._update_ui)
        
        logger.info("Dashboard UI built successfully")
    
    async def _update_ui(self) -> None:
        """Periodic UI update callback."""
        # Process audio events from training loop
        self._process_audio_events()
        
        # Update game frame
        frame = self.state.get_current_frame()
        recent_steps = self.state.get_recent_steps(1)
        action = recent_steps[-1].action if recent_steps else -1
        x_pos = recent_steps[-1].x_pos if recent_steps else 0
        self.game_view.update(frame, action, x_pos)
        
        # Update metrics
        info = self.state.get_training_info()
        self.metrics.update(info)
        
        # Calculate average reward
        episodes = self.state.get_recent_episodes(10)
        if episodes:
            avg_reward = sum(e.total_reward for e in episodes) / len(episodes)
            self.metrics.set_avg_reward(avg_reward)
        
        # Update chart with recent step data
        recent_steps = self.state.get_recent_steps(100)
        if recent_steps:
            # Downsample for chart performance
            step_interval = max(1, len(recent_steps) // 50)
            sampled = recent_steps[::step_interval]
            
            steps = [s.step for s in sampled]
            total_rewards = [s.reward for s in sampled]
            velocity_rewards = [s.velocity_reward for s in sampled]
            clock_penalties = [s.clock_penalty for s in sampled]
            
            self.chart.update(steps, total_rewards, velocity_rewards, clock_penalties)
    
    def _on_start(self) -> None:
        """Handle start button click."""
        logger.info("User clicked START button")
        self.state.start_training()
        self.audio.handle_event(AudioEvent.TRAINING_START)
    
    def _on_pause(self) -> None:
        """Handle pause button click."""
        logger.info("User clicked PAUSE button")
        self.state.pause_training()
    
    def _on_stop(self) -> None:
        """Handle stop button click."""
        logger.info("User clicked STOP button")
        self.state.stop_training()
        self.audio.handle_event(AudioEvent.TRAINING_STOP)
    
    def _on_toggle_mute(self) -> None:
        """Handle mute button click."""
        is_muted = self.audio.toggle_mute()
        self.controls.set_muted(is_muted)
        logger.info(f"Audio muted: {is_muted}")
    
    def _process_audio_events(self) -> None:
        """Process pending audio events from the training loop."""
        events = self.state.pop_audio_events()
        for event_type in events:
            try:
                event = AudioEvent(event_type)
                self.audio.handle_event(event)
            except ValueError:
                logger.warning(f"Unknown audio event: {event_type}")


def create_dashboard(state: Optional[TrainingState] = None) -> TrainingDashboard:
    """Create and build the training dashboard."""
    dashboard = TrainingDashboard(state)
    dashboard.build()
    return dashboard


@ui.page('/')
def main_page():
    """Main dashboard page."""
    create_dashboard()


def run_dashboard(host: str = '127.0.0.1', port: int = 8080) -> None:
    """Run the dashboard server."""
    logger.info(f"Starting dashboard at http://{host}:{port}")
    ui.run(host=host, port=port, title='Mario RL Dashboard', favicon='🎮')


if __name__ == '__main__':
    run_dashboard()
