"""
Explanation Service
Handles explainability visualizations (GradCAM, asymmetry maps, etc.)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from explainability.gradcam import GradCAM
from app.config import GRADCAM_CONFIG


class ExplanationService:
    """Service for explainability operations"""
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize explanation service
        
        Args:
            model: PyTorch model for GradCAM
        """
        self.model = model
        self.gradcam = None
        
        if model is not None:
            target_layer = GRADCAM_CONFIG['target_layer']
            self.gradcam = GradCAM(model, target_layer)
    
    def generate_gradcam(
        self, 
        image_tensor: torch.Tensor, 
        predicted_class: int
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap
        
        Args:
            image_tensor: Preprocessed image tensor
            predicted_class: Predicted class index
            
        Returns:
            GradCAM heatmap as numpy array
        """
        try:
            if self.gradcam is None:
                return np.zeros((224, 224))
            
            heatmap = self.gradcam.generate_cam(image_tensor, predicted_class)
            return heatmap
            
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            return np.zeros((224, 224))
    
    def create_visual_explanations(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
        predicted_class: int,
        left_hemisphere: np.ndarray,
        right_hemisphere: np.ndarray
    ) -> Dict:
        """
        Create all visual explanations
        
        Args:
            image_tensor: Preprocessed image tensor
            original_image: Original image array
            predicted_class: Predicted class
            left_hemisphere: Left hemisphere image
            right_hemisphere: Right hemisphere image
            
        Returns:
            Dictionary of visual explanations
        """
        # Generate GradCAM
        gradcam_heatmap = self.generate_gradcam(image_tensor, predicted_class)
        
        # Create asymmetry map
        asymmetry_map = self._create_asymmetry_map(left_hemisphere, right_hemisphere)
        
        return {
            'gradcam': gradcam_heatmap,
            'gradcam_plus_plus': gradcam_heatmap,  # Using same for now
            'asymmetry_map': asymmetry_map
        }
    
    def _create_asymmetry_map(
        self, 
        left: np.ndarray, 
        right: np.ndarray
    ) -> np.ndarray:
        """
        Create asymmetry heatmap
        
        Args:
            left: Left hemisphere
            right: Right hemisphere (already flipped)
            
        Returns:
            Asymmetry map
        """
        try:
            if left.shape != right.shape:
                return np.zeros((224, 224))
            
            asymmetry = np.abs(left.astype(np.float32) - right.astype(np.float32))
            return asymmetry
            
        except Exception as e:
            print(f"Error creating asymmetry map: {e}")
            return np.zeros((224, 224))
