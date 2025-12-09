"""
Model Service
Handles model loading, preprocessing, and predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import (
    MODEL_CONFIG, 
    PREPROCESSING_CONFIG, 
    CLASS_NAMES,
    get_model_path
)


class PureCNNModel(nn.Module):
    """Pure CNN model - matches training script"""
    
    def __init__(self, num_classes=4, backbone='efficientnet_b3'):
        super(PureCNNModel, self).__init__()
        
        if backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=False)
            backbone_features = 1536
            self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class ModelService:
    """Service for model operations"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load the trained model
        
        Args:
            model_path: Path to model checkpoint (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path is None:
                model_path = get_model_path()
            
            if model_path is None or not model_path.exists():
                return False
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            config = checkpoint.get('config', {})
            model = PureCNNModel(
                num_classes=MODEL_CONFIG['num_classes'],
                backbone=config.get('backbone', MODEL_CONFIG['backbone'])
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.model = model
            self.model_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (tensor for model, array for visualization)
        """
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(MODEL_CONFIG['image_size'])
        
        # Convert to array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array(PREPROCESSING_CONFIG['mean'])
        std = np.array(PREPROCESSING_CONFIG['std'])
        image_normalized = (image_array - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(
            image_normalized.transpose(2, 0, 1)
        ).unsqueeze(0).float()
        
        return image_tensor, image_array
    
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """
        Make prediction on preprocessed image
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Dictionary containing predictions
        """
        if not self.model_loaded or self.model is None:
            raise ValueError("Model not loaded")
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Create uncertainty estimate
        uncertainty = torch.tensor([[1.0 - confidence]])
        
        # Create confidence results
        confidence_results = {
            'predictions': probabilities,
            'std': torch.zeros_like(probabilities),
            'confidence': torch.tensor([[confidence]])
        }
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'confidence_results': confidence_results,
            'class_name': CLASS_NAMES[predicted_class]
        }
    
    def get_model(self) -> Optional[nn.Module]:
        """Get the loaded model"""
        return self.model
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
