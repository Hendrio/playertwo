"""
NiceGUI-based training dashboard for Mario RL agent.

Displays live game view, rewards/penalties chart, and training metrics.
"""

import asyncio
import base64
import io
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from nicegui import ui, app

from .state import TrainingState, get_training_state
from .audio import AudioManager, AudioEvent, get_audio_manager

# Configure dashboard logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "dashboard.log"

logger = logging.getLogger("dashboard")
logger.setLevel(logging.DEBUG)

# File handler with rotation (5MB max, keep 3 backups)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Console handler for warnings and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Dashboard logging initialized")


class TrainingDashboard:
    """
    Real-time training visualization dashboard.
    
    Features:
    - Live game frame display
    - Rewards chart (velocity, clock, death breakdown)
    - Training metrics panel
    - Control buttons
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
        
        # UI elements (will be initialized in build())
        self.game_image: Optional[ui.image] = None
        self.rewards_chart: Optional[ui.chart] = None
        self.metrics_labels: dict = {}
        self.mute_button: Optional[ui.button] = None
        
        # Chart data
        self.reward_history = []
        self.velocity_history = []
        self.clock_history = []
        self.step_history = []
        
        # Update timer
        self.update_interval = 0.2  # 5 FPS (reduced from 10 to minimize flicker)
    
    def build(self):
        """Build the dashboard UI."""
        logger.info("Building dashboard UI")
        
        # Apply dark theme and styling
        ui.dark_mode().enable()
        
        # Add custom CSS
        ui.add_head_html('''
        <style>
            .game-container {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            .metric-card {
                background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
                border-radius: 8px;
                padding: 16px;
                text-align: center;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #e94560;
            }
            .metric-label {
                color: #a0a0a0;
                font-size: 0.9em;
            }
            .chart-container {
                background: #1a1a2e;
                border-radius: 12px;
                padding: 16px;
            }
        </style>
        ''')
        
        with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-4'):
            # Header
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('🎮 Mario RL Training Dashboard').classes('text-3xl font-bold text-white')
                
                with ui.row().classes('gap-2'):
                    # Audio mute button
                    self.mute_button = ui.button('🔊', on_click=self._on_toggle_mute).props('flat')
                    
                    ui.button('▶ Start', on_click=self._on_start).props('color=positive')
                    ui.button('⏸ Pause', on_click=self._on_pause).props('color=warning')
                    ui.button('⏹ Stop', on_click=self._on_stop).props('color=negative')
            
            # Build audio elements (hidden)
            self.audio.build()
            
            # Main content row
            with ui.row().classes('w-full gap-4'):
                # Left column: Game view
                with ui.column().classes('game-container'):
                    ui.label('Live Game View').classes('text-lg font-semibold text-white mb-2')
                    
                    # Placeholder for game frame
                    self.game_image = ui.image().classes('w-96 h-80 rounded-lg')
                    self._update_placeholder_frame()
                    
                    # Action display
                    with ui.row().classes('gap-2 mt-2'):
                        self.action_label = ui.label('Action: -').classes('text-white')
                        self.x_pos_label = ui.label('X: 0').classes('text-white')
                
                # Right column: Metrics
                with ui.column().classes('flex-grow gap-4'):
                    # Metrics cards row
                    with ui.row().classes('w-full gap-4'):
                        with ui.card().classes('metric-card flex-grow'):
                            ui.label('Episode').classes('metric-label')
                            self.metrics_labels['episode'] = ui.label('0').classes('metric-value')
                        
                        with ui.card().classes('metric-card flex-grow'):
                            ui.label('Steps').classes('metric-label')
                            self.metrics_labels['steps'] = ui.label('0').classes('metric-value')
                        
                        with ui.card().classes('metric-card flex-grow'):
                            ui.label('Avg Reward').classes('metric-label')
                            self.metrics_labels['avg_reward'] = ui.label('0.0').classes('metric-value')
                    
                    with ui.row().classes('w-full gap-4'):
                        with ui.card().classes('metric-card flex-grow'):
                            ui.label('Learning Rate').classes('metric-label')
                            self.metrics_labels['lr'] = ui.label('1e-4').classes('metric-value text-xl')
                        
                        with ui.card().classes('metric-card flex-grow'):
                            ui.label('Epsilon').classes('metric-label')
                            self.metrics_labels['epsilon'] = ui.label('1.00').classes('metric-value text-xl')
                        
                        with ui.card().classes('metric-card flex-grow'):
                            ui.label('Status').classes('metric-label')
                            self.metrics_labels['status'] = ui.label('Idle').classes('metric-value text-xl text-green-400')
            
            # Rewards chart
            with ui.card().classes('chart-container w-full'):
                ui.label('📊 Rewards Over Time').classes('text-lg font-semibold text-white mb-2')
                
                self.rewards_chart = ui.echart({
                    'backgroundColor': 'transparent',
                    'tooltip': {'trigger': 'axis'},
                    'legend': {
                        'data': ['Total Reward', 'Velocity', 'Clock'],
                        'textStyle': {'color': '#a0a0a0'}
                    },
                    'xAxis': {
                        'type': 'category',
                        'name': 'Step',
                        'nameTextStyle': {'color': '#a0a0a0'},
                        'axisLabel': {'color': '#a0a0a0'},
                        'axisLine': {'lineStyle': {'color': '#333'}},
                        'data': []
                    },
                    'yAxis': {
                        'type': 'value',
                        'name': 'Reward',
                        'nameTextStyle': {'color': '#a0a0a0'},
                        'axisLabel': {'color': '#a0a0a0'},
                        'axisLine': {'lineStyle': {'color': '#333'}},
                        'splitLine': {'lineStyle': {'color': '#333'}}
                    },
                    'series': [
                        {'name': 'Total Reward', 'type': 'line', 'data': [], 'itemStyle': {'color': '#e94560'}},
                        {'name': 'Velocity', 'type': 'line', 'data': [], 'itemStyle': {'color': '#00d9ff'}},
                        {'name': 'Clock', 'type': 'line', 'data': [], 'itemStyle': {'color': '#ffc107'}},
                    ]
                }).classes('w-full h-64')
            
            # Loss chart
            with ui.card().classes('chart-container w-full'):
                ui.label('📉 Training Losses').classes('text-lg font-semibold text-white mb-2')
                
                with ui.row().classes('w-full gap-8'):
                    with ui.column().classes('flex-grow'):
                        ui.label('Policy Loss').classes('text-gray-400')
                        self.metrics_labels['policy_loss'] = ui.label('0.000').classes('text-2xl text-white')
                    
                    with ui.column().classes('flex-grow'):
                        ui.label('Value Loss').classes('text-gray-400')
                        self.metrics_labels['value_loss'] = ui.label('0.000').classes('text-2xl text-white')
                    
                    with ui.column().classes('flex-grow'):
                        ui.label('Entropy').classes('text-gray-400')
                        self.metrics_labels['entropy'] = ui.label('0.000').classes('text-2xl text-white')
        
        # Start update timer
        ui.timer(self.update_interval, self._update_ui)
    
    def _update_placeholder_frame(self):
        """Display a placeholder when no frame is available."""
        # Create a dark placeholder image
        placeholder = np.zeros((240, 256, 3), dtype=np.uint8)
        placeholder[:] = (26, 26, 46)  # Dark blue background
        
        # Add some text
        cv2.putText(
            placeholder, 
            'Waiting for game...', 
            (40, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (169, 169, 169), 
            2
        )
        
        self._set_image(placeholder)
    
    def _set_image(self, frame: np.ndarray):
        """Convert numpy array to base64 and update image."""
        # Ensure RGB format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Resize for display (2x upscale)
        display_frame = cv2.resize(frame, (384, 336), interpolation=cv2.INTER_NEAREST)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        self.game_image.set_source(f'data:image/jpeg;base64,{b64}')
    
    async def _update_ui(self):
        """Periodic UI update callback."""
        # Process audio events from training loop
        self._process_audio_events()
        
        # Update game frame
        frame = self.state.get_current_frame()
        if frame is not None:
            self._set_image(frame)
        
        # Update metrics
        info = self.state.get_training_info()
        
        self.metrics_labels['episode'].set_text(str(info['episode_count']))
        self.metrics_labels['steps'].set_text(f"{info['global_step']:,}")
        self.metrics_labels['lr'].set_text(f"{info['learning_rate']:.2e}")
        self.metrics_labels['epsilon'].set_text(f"{info['epsilon']:.2f}")
        self.metrics_labels['policy_loss'].set_text(f"{info['policy_loss']:.4f}")
        self.metrics_labels['value_loss'].set_text(f"{info['value_loss']:.4f}")
        self.metrics_labels['entropy'].set_text(f"{info['entropy']:.4f}")
        
        # Update status
        if info['is_training']:
            if info['is_paused']:
                self.metrics_labels['status'].set_text('Paused')
                self.metrics_labels['status'].classes(replace='text-yellow-400')
            else:
                self.metrics_labels['status'].set_text('Training')
                self.metrics_labels['status'].classes(replace='text-green-400')
        else:
            self.metrics_labels['status'].set_text('Idle')
            self.metrics_labels['status'].classes(replace='text-gray-400')
        
        # Update average reward
        recent_episodes = self.state.get_recent_episodes(10)
        if recent_episodes:
            avg_reward = sum(e.total_reward for e in recent_episodes) / len(recent_episodes)
            self.metrics_labels['avg_reward'].set_text(f"{avg_reward:.1f}")
        
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
            
            self.rewards_chart.options['xAxis']['data'] = steps
            self.rewards_chart.options['series'][0]['data'] = total_rewards
            self.rewards_chart.options['series'][1]['data'] = velocity_rewards
            self.rewards_chart.options['series'][2]['data'] = clock_penalties
            self.rewards_chart.update()
    
    def _on_start(self):
        """Handle start button click."""
        logger.info("User clicked START button")
        self.state.start_training()
        self.audio.handle_event(AudioEvent.TRAINING_START)
    
    def _on_pause(self):
        """Handle pause button click."""
        logger.info("User clicked PAUSE button")
        self.state.pause_training()
    
    def _on_stop(self):
        """Handle stop button click."""
        logger.info("User clicked STOP button")
        self.state.stop_training()
        self.audio.handle_event(AudioEvent.TRAINING_STOP)
    
    def _on_toggle_mute(self):
        """Handle mute button click."""
        is_muted = self.audio.toggle_mute()
        icon = '🔇' if is_muted else '🔊'
        self.mute_button.set_text(icon)
        logger.info(f"Audio muted: {is_muted}")
    
    def _process_audio_events(self):
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


def run_dashboard(host: str = '127.0.0.1', port: int = 8080):
    """Run the dashboard server."""
    logger.info(f"Starting dashboard server at http://{host}:{port}")
    ui.run(host=host, port=port, title='Mario RL Training', reload=False)


if __name__ == '__main__':
    run_dashboard()
