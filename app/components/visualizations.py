"""
Visualization Components
Matplotlib-based visualizations for heatmaps and explanations
"""

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import VIZ_CONFIG


class VisualizationComponents:
    """Reusable visualization components"""
    
    @staticmethod
    def create_heatmap_visualization(
        original_image: np.ndarray,
        visual_explanations: Dict[str, np.ndarray]
    ) -> str:
        """
        Create comprehensive heatmap visualization
        
        Args:
            original_image: Original brain scan image
            visual_explanations: Dictionary containing gradcam and asymmetry maps
            
        Returns:
            Base64 encoded image string
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('Explainable AI Visualizations', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # GradCAM
        gradcam = visual_explanations.get('gradcam', np.zeros_like(original_image))
        im1 = axes[1].imshow(gradcam, cmap='jet', alpha=0.7)
        axes[1].imshow(original_image, cmap='gray', alpha=0.3)
        axes[1].set_title('GradCAM')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # GradCAM++
        gradcam_pp = visual_explanations.get('gradcam_plus_plus', gradcam)
        im2 = axes[2].imshow(gradcam_pp, cmap='jet', alpha=0.7)
        axes[2].imshow(original_image, cmap='gray', alpha=0.3)
        axes[2].set_title('GradCAM++')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Asymmetry map
        asymmetry_map = visual_explanations.get('asymmetry_map', np.zeros_like(original_image))
        im3 = axes[3].imshow(asymmetry_map, cmap='hot')
        axes[3].set_title('Asymmetry Map')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    @staticmethod
    def create_hemisphere_comparison(
        left_hemisphere: np.ndarray,
        right_hemisphere: np.ndarray,
        midline: int,
        original_image: np.ndarray
    ) -> str:
        """
        Create hemisphere comparison visualization
        
        Args:
            left_hemisphere: Left hemisphere image
            right_hemisphere: Right hemisphere image (flipped)
            midline: Detected midline position
            original_image: Original image
            
        Returns:
            Base64 encoded image string
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('Hemisphere Symmetry Analysis', fontsize=16, fontweight='bold')
        
        # Original with midline
        axes[0].imshow(original_image, cmap='gray')
        axes[0].axvline(x=midline, color='red', linewidth=2, label='Midline')
        axes[0].set_title('Original with Midline')
        axes[0].legend()
        axes[0].axis('off')
        
        # Left hemisphere
        axes[1].imshow(left_hemisphere, cmap='gray')
        axes[1].set_title('Left Hemisphere')
        axes[1].axis('off')
        
        # Right hemisphere (flipped)
        axes[2].imshow(right_hemisphere, cmap='gray')
        axes[2].set_title('Right Hemisphere (Flipped)')
        axes[2].axis('off')
        
        # Difference map
        if left_hemisphere.shape == right_hemisphere.shape:
            diff_map = np.abs(left_hemisphere.astype(np.float32) - 
                            right_hemisphere.astype(np.float32))
            im = axes[3].imshow(diff_map, cmap='hot')
            axes[3].set_title('Difference Map')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    @staticmethod
    def create_single_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        title: str = "Heatmap Overlay",
        cmap: str = 'jet',
        alpha: float = 0.6
    ) -> str:
        """
        Create single heatmap overlay
        
        Args:
            image: Base image
            heatmap: Heatmap to overlay
            title: Plot title
            cmap: Colormap name
            alpha: Overlay transparency
            
        Returns:
            Base64 encoded image string
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Show base image
        ax.imshow(image, cmap='gray')
        
        # Overlay heatmap
        im = ax.imshow(heatmap, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
