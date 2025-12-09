"""
Chart Components
Plotly chart generation for predictions and symmetry analysis
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import CLASS_NAMES, VIZ_CONFIG


class ChartComponents:
    """Reusable chart components"""
    
    @staticmethod
    def create_prediction_chart(probabilities: np.ndarray) -> go.Figure:
        """
        Create prediction probability bar chart
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[
            go.Bar(
                x=CLASS_NAMES,
                y=probabilities,
                marker_color=[
                    'red' if i == np.argmax(probabilities) else 'lightblue'
                    for i in range(len(probabilities))
                ],
                text=[f'{p:.3f}' for p in probabilities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Prediction Probabilities',
            xaxis_title='Tumor Class',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_symmetry_chart(symmetry_features: Dict[str, float]) -> go.Figure:
        """
        Create symmetry features horizontal bar chart
        
        Args:
            symmetry_features: Dictionary of symmetry metrics
            
        Returns:
            Plotly figure
        """
        # Format metric names for display
        feature_names = [
            name.replace('_', ' ').replace('hemisphere ', '').title() 
            for name in symmetry_features.keys()
        ]
        feature_values = list(symmetry_features.values())
        
        fig = go.Figure(data=[
            go.Bar(
                y=feature_names,
                x=feature_values,
                orientation='h',
                marker_color='skyblue',
                text=[f'{v:.3f}' for v in feature_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Brain Symmetry Analysis (Core 4 Metrics)',
            xaxis_title='Score (0-1)',
            yaxis_title='Symmetry Metrics',
            xaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_confidence_chart(confidence_data: Dict) -> go.Figure:
        """
        Create confidence interval chart
        
        Args:
            confidence_data: Dictionary with mean, std, confidence
            
        Returns:
            Plotly figure
        """
        mean_probs = confidence_data['mean']
        std_probs = confidence_data['std']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=CLASS_NAMES,
            y=mean_probs,
            error_y=dict(type='data', array=std_probs),
            marker_color='lightgreen',
            name='Mean Probability'
        ))
        
        fig.update_layout(
            title='Prediction Confidence Intervals',
            xaxis_title='Tumor Class',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_symmetry_radar_chart(symmetry_features: Dict[str, float]) -> go.Figure:
        """
        Create radar chart for symmetry metrics
        
        Args:
            symmetry_features: Dictionary of symmetry metrics
            
        Returns:
            Plotly figure
        """
        # Format names
        feature_names = [
            name.replace('_', ' ').replace('hemisphere ', '').title()
            for name in symmetry_features.keys()
        ]
        feature_values = list(symmetry_features.values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=feature_values,
            theta=feature_names,
            fill='toself',
            marker_color='rgba(31, 119, 180, 0.6)',
            line_color='rgba(31, 119, 180, 1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Symmetry Metrics Radar Chart',
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(
        metric_name: str,
        values: List[float],
        labels: List[str]
    ) -> go.Figure:
        """
        Create comparison bar chart
        
        Args:
            metric_name: Name of the metric
            values: List of values
            labels: List of labels
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=VIZ_CONFIG['primary_color'],
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'{metric_name} Comparison',
            xaxis_title='Category',
            yaxis_title=metric_name,
            height=400,
            showlegend=False
        )
        
        return fig
