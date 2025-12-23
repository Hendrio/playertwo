"""
Rewards chart component for visualizing training rewards.
"""

from typing import List, Optional
from nicegui import ui

from ..styles import CHART_COLORS


class RewardsChart:
    """
    ECharts-based rewards visualization.
    
    Shows:
    - Total reward over time
    - Velocity reward component
    - Clock penalty component
    """
    
    def __init__(self):
        """Initialize the rewards chart."""
        self.chart: Optional[ui.echart] = None
    
    def build(self) -> None:
        """Build the rewards chart UI element."""
        with ui.card().classes('chart-container w-full'):
            ui.label('📊 Rewards Over Time').classes('text-lg font-semibold text-white mb-2')
            
            self.chart = ui.echart({
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
                    {
                        'name': 'Total Reward',
                        'type': 'line',
                        'smooth': True,
                        'data': [],
                        'lineStyle': {'color': CHART_COLORS['total_reward']},
                        'itemStyle': {'color': CHART_COLORS['total_reward']}
                    },
                    {
                        'name': 'Velocity',
                        'type': 'line',
                        'smooth': True,
                        'data': [],
                        'lineStyle': {'color': CHART_COLORS['velocity']},
                        'itemStyle': {'color': CHART_COLORS['velocity']}
                    },
                    {
                        'name': 'Clock',
                        'type': 'line',
                        'smooth': True,
                        'data': [],
                        'lineStyle': {'color': CHART_COLORS['clock']},
                        'itemStyle': {'color': CHART_COLORS['clock']}
                    }
                ],
                'grid': {
                    'left': '10%',
                    'right': '5%',
                    'bottom': '15%',
                    'top': '20%'
                }
            }).classes('w-full h-64')
    
    def update(
        self,
        steps: List[int],
        total_rewards: List[float],
        velocity_rewards: List[float],
        clock_penalties: List[float]
    ) -> None:
        """
        Update the chart with new data.
        
        Args:
            steps: List of step numbers
            total_rewards: List of total rewards
            velocity_rewards: List of velocity rewards
            clock_penalties: List of clock penalties
        """
        if self.chart:
            self.chart.options['xAxis']['data'] = steps
            self.chart.options['series'][0]['data'] = total_rewards
            self.chart.options['series'][1]['data'] = velocity_rewards
            self.chart.options['series'][2]['data'] = clock_penalties
            self.chart.update()
