"""
Live game view component for displaying Mario gameplay frames.
"""

import base64
import cv2
import numpy as np
from typing import Optional
from nicegui import ui


class GameViewComponent:
    """
    Displays live game frames from the training environment.
    
    Features:
    - Real-time frame updates
    - Placeholder when no frames available
    - Action and position display
    """
    
    def __init__(self):
        """Initialize the game view component."""
        self.game_image: Optional[ui.image] = None
        self.action_label: Optional[ui.label] = None
        self.x_pos_label: Optional[ui.label] = None
    
    def build(self) -> None:
        """Build the game view UI elements."""
        with ui.column().classes('game-container'):
            ui.label('Live Game View').classes('text-lg font-semibold text-white mb-2')
            
            # Game frame display
            self.game_image = ui.image().classes('w-96 h-80 rounded-lg')
            self._show_placeholder()
            
            # Action and position display
            with ui.row().classes('gap-2 mt-2'):
                self.action_label = ui.label('Action: -').classes('text-white')
                self.x_pos_label = ui.label('X: 0').classes('text-white')
    
    def _show_placeholder(self) -> None:
        """Display placeholder when no frame is available."""
        # Create a dark placeholder image
        placeholder = np.zeros((240, 256, 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 40)  # Dark gray
        
        # Add "Waiting for game..." text
        cv2.putText(
            placeholder, "Waiting for game...",
            (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2
        )
        
        self._set_frame(placeholder)
    
    def _set_frame(self, frame: np.ndarray) -> None:
        """Convert numpy array to base64 and update image."""
        # Handle grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Resize for display (2x upscale)
        display_frame = cv2.resize(frame, (384, 336), interpolation=cv2.INTER_NEAREST)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        if self.game_image:
            self.game_image.set_source(f'data:image/jpeg;base64,{b64}')
    
    def update(self, frame: Optional[np.ndarray], action: int = -1, x_pos: int = 0) -> None:
        """
        Update the game view with new frame and stats.
        
        Args:
            frame: RGB frame array or None for placeholder
            action: Current action index
            x_pos: Mario's x position
        """
        if frame is not None:
            self._set_frame(frame)
        
        if self.action_label:
            action_names = ['Right', 'Right+Jump', 'Right+Run', 'Jump']
            action_str = action_names[action] if 0 <= action < len(action_names) else '-'
            self.action_label.set_text(f'Action: {action_str}')
        
        if self.x_pos_label:
            self.x_pos_label.set_text(f'X: {x_pos}')
