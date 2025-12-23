"""
Metrics panel component for displaying training statistics.
"""

from typing import Dict, Optional
from nicegui import ui


class MetricsPanel:
    """
    Displays training metrics in card format.
    
    Metrics shown:
    - Episode count
    - Total steps
    - Average reward
    - Learning rate
    - Epsilon
    - Status
    - Policy loss, Value loss, Entropy
    """
    
    def __init__(self):
        """Initialize the metrics panel."""
        self.labels: Dict[str, ui.label] = {}
    
    def build(self) -> None:
        """Build the metrics panel UI elements."""
        with ui.column().classes('flex-grow gap-4'):
            # Top row: Episode, Steps, Avg Reward
            with ui.row().classes('w-full gap-4'):
                self._create_metric_card('episode', 'Episode', '0')
                self._create_metric_card('steps', 'Steps', '0')
                self._create_metric_card('avg_reward', 'Avg Reward', '0.0')
            
            # Middle row: LR, Epsilon, Status
            with ui.row().classes('w-full gap-4'):
                self._create_metric_card('lr', 'Learning Rate', '1e-4', 'text-xl')
                self._create_metric_card('epsilon', 'Epsilon', '1.00', 'text-xl')
                self._create_metric_card('status', 'Status', 'Idle', 'text-xl text-green-400')
            
            # Bottom row: Losses
            with ui.row().classes('w-full gap-4'):
                self._create_metric_card('policy_loss', 'Policy Loss', '0.0000', 'text-xl')
                self._create_metric_card('value_loss', 'Value Loss', '0.0000', 'text-xl')
                self._create_metric_card('entropy', 'Entropy', '0.0000', 'text-xl')
    
    def _create_metric_card(self, key: str, label: str, initial_value: str, 
                           value_classes: str = '') -> None:
        """Create a single metric card."""
        with ui.card().classes('metric-card flex-grow'):
            ui.label(label).classes('metric-label')
            self.labels[key] = ui.label(initial_value).classes(f'metric-value {value_classes}')
    
    def update(self, info: Dict) -> None:
        """
        Update metrics from training info dict.
        
        Args:
            info: Dictionary with training info from TrainingState
        """
        if 'episode_count' in info:
            self.labels['episode'].set_text(str(info['episode_count']))
        
        if 'global_step' in info:
            self.labels['steps'].set_text(f"{info['global_step']:,}")
        
        if 'learning_rate' in info:
            self.labels['lr'].set_text(f"{info['learning_rate']:.2e}")
        
        if 'epsilon' in info:
            self.labels['epsilon'].set_text(f"{info['epsilon']:.2f}")
        
        if 'policy_loss' in info:
            self.labels['policy_loss'].set_text(f"{info['policy_loss']:.4f}")
        
        if 'value_loss' in info:
            self.labels['value_loss'].set_text(f"{info['value_loss']:.4f}")
        
        if 'entropy' in info:
            self.labels['entropy'].set_text(f"{info['entropy']:.4f}")
        
        # Update status
        if info.get('is_training'):
            status = 'Paused' if info.get('is_paused') else 'Training'
            color = 'text-yellow-400' if info.get('is_paused') else 'text-green-400'
        else:
            status = 'Idle'
            color = 'text-gray-400'
        
        self.labels['status'].set_text(status)
        self.labels['status'].classes(replace=f'metric-value text-xl {color}')
    
    def set_avg_reward(self, avg_reward: float) -> None:
        """Set the average reward value."""
        self.labels['avg_reward'].set_text(f"{avg_reward:.1f}")
