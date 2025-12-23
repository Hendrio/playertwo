"""
CSS styles for the Mario RL training dashboard.

Separates styling from component logic for better maintainability.
"""

# Dark theme CSS for the dashboard
DASHBOARD_CSS = '''
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
    .control-button {
        min-width: 100px;
    }
    .header-title {
        font-size: 1.875rem;
        font-weight: bold;
        color: white;
    }
</style>
'''

# Color palette
COLORS = {
    'primary': '#e94560',
    'secondary': '#0f3460',
    'background': '#1a1a2e',
    'background_alt': '#16213e',
    'text': '#ffffff',
    'text_muted': '#a0a0a0',
    'success': '#4caf50',
    'warning': '#ff9800',
    'error': '#f44336',
}

# Chart color scheme
CHART_COLORS = {
    'total_reward': '#e94560',
    'velocity': '#4ecdc4',
    'clock': '#ffe66d',
}
