"""
Symmetry-Integrated CNN for Brain Tumor Classification
Combines CNN visual features with brain symmetry metrics for enhanced diagnosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .symmetry_analyzer import BrainSymmetryAnalyzer


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module - focuses on 'WHERE' to look in the image
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(attention_map)
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention
        return x * attention_map


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module - focuses on 'WHAT' features are important
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling path
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        
        # Max pooling path
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out)
        
        # Combine and apply attention
        attention_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention_weights


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features from multiple scales of the CNN backbone
    """
    def __init__(self, backbone):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.backbone = backbone
        
        # Hook to extract intermediate features
        self.features = {}
        self.hooks = []
        
        # Register hooks for different layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # For EfficientNet, extract features from different blocks
        if hasattr(self.backbone, 'features'):
            # EfficientNet structure
            for i, layer in enumerate(self.backbone.features):
                if i in [3, 5, 7]:  # Extract from multiple scales
                    hook = layer.register_forward_hook(get_activation(f'scale_{i}'))
                    self.hooks.append(hook)
    
    def forward(self, x):
        # Clear previous features
        self.features.clear()
        
        # Forward pass through backbone
        output = self.backbone(x)
        
        return output, self.features
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()


class SymmetryIntegratedCNN(nn.Module):
    """
    Main model that integrates CNN visual features with brain symmetry metrics
    """
    def __init__(self, num_classes=4, backbone='efficientnet_b3', pretrained=True, 
                 freeze_backbone=False, symmetry_weight=0.3):
        super(SymmetryIntegratedCNN, self).__init__()
        
        self.num_classes = num_classes
        self.symmetry_weight = symmetry_weight
        
        # Initialize symmetry analyzer
        self.symmetry_analyzer = BrainSymmetryAnalyzer(image_size=(224, 224))
        
        # CNN Backbone
        self.backbone = self._create_backbone(backbone, pretrained, freeze_backbone)
        
        # Get backbone output features
        self.backbone_features = self._get_backbone_features(backbone)
        
        # Attention modules
        self.channel_attention = ChannelAttentionModule(self.backbone_features)
        self.spatial_attention = SpatialAttentionModule()
        
        # Multi-scale feature extraction
        self.multi_scale_extractor = MultiScaleFeatureExtractor(self.backbone)
        
        # Symmetry feature processing
        self.symmetry_features = 8  # Number of symmetry metrics
        self.symmetry_processor = nn.Sequential(
            nn.Linear(self.symmetry_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.backbone_features + 128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty estimation (for confidence scoring)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self, backbone_name, pretrained, freeze_backbone):
        """Create CNN backbone"""
        if backbone_name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(pretrained=pretrained)
            # Remove the final classifier
            backbone.classifier = nn.Identity()
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            backbone.fc = nn.Identity()
        elif backbone_name == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        return backbone
    
    def _get_backbone_features(self, backbone_name):
        """Get the number of output features from backbone"""
        if backbone_name == 'efficientnet_b3':
            return 1536
        elif backbone_name == 'resnet50':
            return 2048
        elif backbone_name == 'densenet121':
            return 1024
        else:
            return 1536  # Default
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def extract_symmetry_features(self, images):
        """
        Extract symmetry features from batch of images
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Symmetry features tensor (B, symmetry_features)
        """
        batch_size = images.shape[0]
        symmetry_features = []
        
        for i in range(batch_size):
            # Convert tensor to numpy
            img = images[i].cpu().numpy()
            
            # Handle different input formats
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
            elif img.shape[0] == 1:  # Grayscale
                img = img[0]
            
            # Extract symmetry features
            features = self.symmetry_analyzer.extract_all_symmetry_features(img)
            
            # Convert to list in consistent order
            feature_vector = [
                features['intensity_symmetry'],
                features['texture_symmetry'],
                features['structural_symmetry'],
                features['statistical_symmetry'],
                features['volume_asymmetry'],
                features['midline_position'],
                features['hemisphere_correlation'],
                features['asymmetry_index']
            ]
            
            symmetry_features.append(feature_vector)
        
        return torch.tensor(symmetry_features, dtype=torch.float32, device=images.device)
    
    def forward(self, x, return_features=False):
        """
        Forward pass through the integrated model
        
        Args:
            x: Input images (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Classification logits and optionally intermediate features
        """
        batch_size = x.shape[0]
        
        # CNN Branch - Visual Features
        cnn_features, multi_scale_features = self.multi_scale_extractor(x)
        
        # Apply attention mechanisms
        cnn_features = cnn_features.view(batch_size, self.backbone_features, 1, 1)
        cnn_features = self.channel_attention(cnn_features)
        cnn_features = self.spatial_attention(cnn_features)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).view(batch_size, -1)
        
        # Symmetry Branch - Quantitative Features
        symmetry_features = self.extract_symmetry_features(x)
        symmetry_processed = self.symmetry_processor(symmetry_features)
        
        # Feature Fusion
        combined_features = torch.cat([cnn_features, symmetry_processed], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(fused_features)
        
        if return_features:
            return {
                'logits': logits,
                'uncertainty': uncertainty,
                'cnn_features': cnn_features,
                'symmetry_features': symmetry_features,
                'fused_features': fused_features
            }
        
        return logits, uncertainty
    
    def predict_with_confidence(self, x, num_samples=10):
        """
        Predict with uncertainty estimation using Monte Carlo Dropout
        
        Args:
            x: Input images
            num_samples: Number of MC samples
            
        Returns:
            Predictions with confidence intervals
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits, uncertainty = self.forward(x)
                predictions.append(F.softmax(logits, dim=1))
                uncertainties.append(uncertainty)
        
        # Calculate statistics
        predictions = torch.stack(predictions)
        uncertainties = torch.stack(uncertainties)
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        mean_uncertainty = torch.mean(uncertainties, dim=0)
        
        # Calculate confidence (1 - uncertainty)
        confidence = 1.0 - mean_uncertainty
        
        return {
            'predictions': mean_pred,
            'std': std_pred,
            'confidence': confidence,
            'uncertainty': mean_uncertainty
        }
    
    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization
        
        Args:
            x: Input images
            
        Returns:
            Dictionary of attention maps
        """
        # Forward pass with hooks to capture attention
        with torch.no_grad():
            features = self.forward(x, return_features=True)
        
        # Extract attention weights (this would need modification based on actual implementation)
        attention_maps = {
            'channel_attention': None,  # Would extract from channel attention module
            'spatial_attention': None,  # Would extract from spatial attention module
            'symmetry_contribution': features['symmetry_features']
        }
        
        return attention_maps
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_features': self.backbone_features,
            'symmetry_features': self.symmetry_features,
            'num_classes': self.num_classes
        }


class SymmetryLoss(nn.Module):
    """
    Custom loss function that combines classification loss with symmetry consistency
    """
    def __init__(self, alpha=1.0, beta=0.1):
        super(SymmetryLoss, self).__init__()
        self.alpha = alpha  # Classification loss weight
        self.beta = beta    # Symmetry consistency weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, logits, targets, symmetry_features, uncertainty=None):
        """
        Compute combined loss
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            symmetry_features: Extracted symmetry features
            uncertainty: Uncertainty estimates (optional)
            
        Returns:
            Combined loss
        """
        # Classification loss
        classification_loss = self.ce_loss(logits, targets)
        
        # Symmetry consistency loss (encourage consistent symmetry patterns)
        # This is a placeholder - would implement based on specific requirements
        symmetry_loss = torch.tensor(0.0, device=logits.device)
        
        # Uncertainty loss (optional)
        uncertainty_loss = torch.tensor(0.0, device=logits.device)
        if uncertainty is not None:
            # Encourage lower uncertainty for correct predictions
            probs = F.softmax(logits, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            correct_mask = (predicted == targets).float()
            uncertainty_loss = self.mse_loss(uncertainty.squeeze(), 1.0 - correct_mask)
        
        # Combine losses
        total_loss = (self.alpha * classification_loss + 
                     self.beta * symmetry_loss + 
                     0.1 * uncertainty_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'symmetry_loss': symmetry_loss,
            'uncertainty_loss': uncertainty_loss
        }


# Model factory function
def create_symmetry_cnn(num_classes=4, backbone='efficientnet_b3', pretrained=True, **kwargs):
    """
    Factory function to create SymmetryIntegratedCNN model
    
    Args:
        num_classes: Number of output classes
        backbone: CNN backbone architecture
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments
        
    Returns:
        SymmetryIntegratedCNN model
    """
    model = SymmetryIntegratedCNN(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_symmetry_cnn(num_classes=4)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"{key}: {value:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        logits, uncertainty = model(dummy_input)
        print(f"\nOutput shapes:")
        print(f"Logits: {logits.shape}")
        print(f"Uncertainty: {uncertainty.shape}")
        
        # Test prediction with confidence
        results = model.predict_with_confidence(dummy_input, num_samples=5)
        print(f"Predictions: {results['predictions'].shape}")
        print(f"Confidence: {results['confidence'].shape}")
