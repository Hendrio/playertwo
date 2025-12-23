"""
Control buttons component for training controls.
"""

from typing import Callable, Optional
from nicegui import ui


class ControlButtons:
    """
    Control buttons for training: Start, Pause, Stop, Mute.
    
    Each button triggers a callback when clicked.
    """
    
    def __init__(
        self,
        on_start: Optional[Callable] = None,
        on_pause: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
        on_toggle_mute: Optional[Callable] = None
    ):
        """
        Initialize control buttons.
        
        Args:
            on_start: Callback when Start is clicked
            on_pause: Callback when Pause is clicked
            on_stop: Callback when Stop is clicked
            on_toggle_mute: Callback when Mute is toggled
        """
        self._on_start = on_start
        self._on_pause = on_pause
        self._on_stop = on_stop
        self._on_toggle_mute = on_toggle_mute
        
        self.mute_button: Optional[ui.button] = None
        self._is_muted = False
    
    def build(self) -> None:
        """Build the control buttons UI elements."""
        with ui.row().classes('gap-2'):
            # Audio mute button
            self.mute_button = ui.button('🔊', on_click=self._handle_mute).props('flat')
            
            # Training controls
            ui.button('▶ Start', on_click=self._handle_start).props('color=positive')
            ui.button('⏸ Pause', on_click=self._handle_pause).props('color=warning')
            ui.button('⏹ Stop', on_click=self._handle_stop).props('color=negative')
    
    def _handle_start(self) -> None:
        """Handle start button click."""
        if self._on_start:
            self._on_start()
    
    def _handle_pause(self) -> None:
        """Handle pause button click."""
        if self._on_pause:
            self._on_pause()
    
    def _handle_stop(self) -> None:
        """Handle stop button click."""
        if self._on_stop:
            self._on_stop()
    
    def _handle_mute(self) -> None:
        """Handle mute button click."""
        self._is_muted = not self._is_muted
        icon = '🔇' if self._is_muted else '🔊'
        if self.mute_button:
            self.mute_button.set_text(icon)
        
        if self._on_toggle_mute:
            self._on_toggle_mute()
    
    def set_muted(self, muted: bool) -> None:
        """Set the mute state externally."""
        self._is_muted = muted
        icon = '🔇' if muted else '🔊'
        if self.mute_button:
            self.mute_button.set_text(icon)
