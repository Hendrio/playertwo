"""
Audio manager for the Mario RL training dashboard.

Provides background music and sound effects for the training UI.
Uses NiceGUI audio elements for browser-based playback.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from nicegui import ui

logger = logging.getLogger("dashboard")


class AudioEvent(Enum):
    """Audio event types triggered by game state."""
    TRAINING_START = "training_start"
    TRAINING_STOP = "training_stop"
    DEATH = "death"
    LEVEL_COMPLETE = "level_complete"
    COIN = "coin"


@dataclass
class AudioConfig:
    """Configuration for audio settings."""
    music_volume: float = 0.3
    sfx_volume: float = 0.5
    music_enabled: bool = True
    sfx_enabled: bool = True


class AudioManager:
    """
    Manages audio playback for the training dashboard.
    
    Features:
    - Background music with loop support
    - Sound effects triggered by game events
    - Mute/unmute controls
    - Browser-based playback via NiceGUI
    """
    
    # Asset paths relative to project root
    ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "audio"
    
    # Audio file names
    MUSIC_FILE = "music_theme.mp3"
    SFX_FILES = {
        AudioEvent.DEATH: "sfx_death.mp3",
        AudioEvent.LEVEL_COMPLETE: "sfx_level_complete.mp3",
        AudioEvent.COIN: "sfx_coin.mp3",
    }
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize the audio manager.
        
        Args:
            config: Audio configuration settings
        """
        self.config = config or AudioConfig()
        self._muted = False
        
        # NiceGUI audio elements (initialized when build() is called)
        self._music_element: Optional[ui.audio] = None
        self._sfx_elements: dict = {}
        
        # Callbacks
        self._on_mute_change: Optional[Callable[[bool], None]] = None
        
        logger.info("AudioManager initialized")
    
    def build(self) -> None:
        """
        Create the audio elements in the UI.
        
        Must be called within a NiceGUI page context.
        """
        # Create music element (hidden, auto-loop)
        music_path = self.ASSETS_DIR / self.MUSIC_FILE
        if music_path.exists():
            self._music_element = ui.audio(str(music_path)).props('loop').classes('hidden')
            self._music_element.props(f'volume={self.config.music_volume}')
            logger.info(f"Music element created: {music_path}")
        else:
            logger.warning(f"Music file not found: {music_path}")
        
        # Create SFX elements (hidden)
        for event, filename in self.SFX_FILES.items():
            sfx_path = self.ASSETS_DIR / filename
            if sfx_path.exists():
                element = ui.audio(str(sfx_path)).classes('hidden')
                element.props(f'volume={self.config.sfx_volume}')
                self._sfx_elements[event] = element
                logger.info(f"SFX element created: {sfx_path}")
            else:
                logger.debug(f"SFX file not found: {sfx_path}")
    
    def play_music(self) -> None:
        """Start playing background music."""
        if self._muted or not self.config.music_enabled:
            return
        
        if self._music_element:
            self._music_element.play()
            logger.debug("Music started")
    
    def stop_music(self) -> None:
        """Stop background music."""
        if self._music_element:
            self._music_element.pause()
            logger.debug("Music stopped")
    
    def play_sfx(self, event: AudioEvent) -> None:
        """
        Play a sound effect for the given event.
        
        Args:
            event: The audio event type
        """
        if self._muted or not self.config.sfx_enabled:
            return
        
        element = self._sfx_elements.get(event)
        if element:
            # Reset to start and play
            element.seek(0)
            element.play()
            logger.debug(f"SFX played: {event.value}")
    
    def set_muted(self, muted: bool) -> None:
        """
        Set mute state for all audio.
        
        Args:
            muted: True to mute, False to unmute
        """
        self._muted = muted
        
        if muted:
            self.stop_music()
        
        if self._on_mute_change:
            self._on_mute_change(muted)
        
        logger.info(f"Audio muted: {muted}")
    
    def toggle_mute(self) -> bool:
        """
        Toggle mute state.
        
        Returns:
            New mute state
        """
        self.set_muted(not self._muted)
        return self._muted
    
    def is_muted(self) -> bool:
        """Check if audio is muted."""
        return self._muted
    
    def on_mute_change(self, callback: Callable[[bool], None]) -> None:
        """
        Register callback for mute state changes.
        
        Args:
            callback: Function called with new mute state
        """
        self._on_mute_change = callback
    
    def handle_event(self, event: AudioEvent) -> None:
        """
        Handle an audio event from the game.
        
        Args:
            event: The audio event to handle
        """
        if event == AudioEvent.TRAINING_START:
            self.play_music()
        elif event == AudioEvent.TRAINING_STOP:
            self.stop_music()
        elif event in self.SFX_FILES:
            self.play_sfx(event)


# Global audio manager instance
_audio_manager: Optional[AudioManager] = None


def get_audio_manager() -> AudioManager:
    """Get or create the global audio manager."""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioManager()
    return _audio_manager


def reset_audio_manager() -> None:
    """Reset the global audio manager."""
    global _audio_manager
    _audio_manager = AudioManager()
