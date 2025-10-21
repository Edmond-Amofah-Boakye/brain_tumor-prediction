"""
GradCAM and Explainable AI Module for Brain Tumor Classification
Provides visual explanations for model predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (GradCAM) implementation
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize GradCAM
        
        Args:
            model: PyTorch model
            target_layer: Name of the target layer for activation extraction
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class SymmetryExplainer:
    """
    Explainer for symmetry-based features
    """
    
    def __init__(self, symmetry_analyzer):
        """
        Initialize symmetry explainer
        
        Args:
            symmetry_analyzer: BrainSymmetryAnalyzer instance
        """
        self.symmetry_analyzer = symmetry_analyzer
    
    def explain_symmetry_features(self, image: np.ndarray, features: Dict[str, float]) -> Dict:
        """
        Generate explanations for symmetry features
        
        Args:
            image: Input brain image
            features: Symmetry features dictionary
            
        Returns:
            Dictionary with explanations and visualizations
        """
        # Extract symmetry features and store intermediate results
        _ = self.symmetry_analyzer.extract_all_symmetry_features(image)
        
        explanations = {
            'feature_importance': self._calculate_feature_importance(features),
            'asymmetry_map': self._generate_asymmetry_map(),
            'midline_analysis': self._analyze_midline_deviation(),
            'hemisphere_comparison': self._compare_hemispheres(),
            'clinical_interpretation': self._generate_clinical_interpretation(features)
        }
        
        return explanations
    
    def _calculate_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate relative importance of symmetry features"""
        # Normalize features to [0, 1] and calculate importance
        feature_values = list(features.values())
        max_val = max(feature_values) if feature_values else 1.0
        
        importance = {}
        for feature_name, value in features.items():
            # Higher asymmetry (lower symmetry) = higher importance
            if 'asymmetry' in feature_name:
                importance[feature_name] = value / max_val
            else:
                importance[feature_name] = (1.0 - value) / max_val
        
        return importance
    
    def _generate_asymmetry_map(self) -> np.ndarray:
        """Generate asymmetry heatmap"""
        if hasattr(self.symmetry_analyzer, 'left_hemisphere') and \
           hasattr(self.symmetry_analyzer, 'right_hemisphere'):
            
            left = self.symmetry_analyzer.left_hemisphere
            right = self.symmetry_analyzer.right_hemisphere
            
            if left.shape == right.shape:
                asymmetry_map = np.abs(left.astype(np.float32) - right.astype(np.float32))
                return asymmetry_map / asymmetry_map.max() if asymmetry_map.max() > 0 else asymmetry_map
        
        return np.zeros((224, 224))
    
    def _analyze_midline_deviation(self) -> Dict:
        """Analyze midline deviation from center"""
        if hasattr(self.symmetry_analyzer, 'midline') and \
           hasattr(self.symmetry_analyzer, 'processed_image'):
            
            image_width = self.symmetry_analyzer.processed_image.shape[1]
            center = image_width // 2
            midline = self.symmetry_analyzer.midline
            
            deviation = abs(midline - center)
            deviation_percentage = (deviation / center) * 100
            
            return {
                'midline_position': midline,
                'center_position': center,
                'deviation_pixels': deviation,
                'deviation_percentage': deviation_percentage,
                'interpretation': self._interpret_midline_deviation(deviation_percentage)
            }
        
        return {}
    
    def _interpret_midline_deviation(self, deviation_percentage: float) -> str:
        """Interpret midline deviation"""
        if deviation_percentage < 5:
            return "Normal midline position"
        elif deviation_percentage < 10:
            return "Slight midline shift - may indicate mild asymmetry"
        elif deviation_percentage < 20:
            return "Moderate midline shift - suggests structural asymmetry"
        else:
            return "Significant midline shift - indicates major structural changes"
    
    def _compare_hemispheres(self) -> Dict:
        """Compare left and right hemispheres"""
        if hasattr(self.symmetry_analyzer, 'left_hemisphere') and \
           hasattr(self.symmetry_analyzer, 'right_hemisphere'):
            
            left = self.symmetry_analyzer.left_hemisphere
            right = self.symmetry_analyzer.right_hemisphere
            
            if left.shape == right.shape:
                left_stats = {
                    'mean_intensity': np.mean(left),
                    'std_intensity': np.std(left),
                    'max_intensity': np.max(left),
                    'min_intensity': np.min(left)
                }
                
                right_stats = {
                    'mean_intensity': np.mean(right),
                    'std_intensity': np.std(right),
                    'max_intensity': np.max(right),
                    'min_intensity': np.min(right)
                }
                
                differences = {
                    'mean_diff': abs(left_stats['mean_intensity'] - right_stats['mean_intensity']),
                    'std_diff': abs(left_stats['std_intensity'] - right_stats['std_intensity']),
                    'max_diff': abs(left_stats['max_intensity'] - right_stats['max_intensity']),
                    'min_diff': abs(left_stats['min_intensity'] - right_stats['min_intensity'])
                }
                
                return {
                    'left_hemisphere': left_stats,
                    'right_hemisphere': right_stats,
                    'differences': differences
                }
        
        return {}
    
    def _generate_clinical_interpretation(self, features: Dict[str, float]) -> str:
        """Generate clinical interpretation of symmetry analysis"""
        interpretations = []
        
        # Analyze overall symmetry
        overall_symmetry = np.mean(list(features.values()))
        if overall_symmetry > 0.8:
            interpretations.append("High brain symmetry - consistent with normal anatomy")
        elif overall_symmetry > 0.6:
            interpretations.append("Moderate brain symmetry - some asymmetric features detected")
        else:
            interpretations.append("Low brain symmetry - significant asymmetric features present")
        
        # Analyze specific features
        if features.get('intensity_symmetry', 0) < 0.5:
            interpretations.append("Significant intensity asymmetry detected")
        
        if features.get('structural_symmetry', 0) < 0.5:
            interpretations.append("Structural asymmetry observed")
        
        if features.get('volume_asymmetry', 0) < 0.5:
            interpretations.append("Volume differences between hemispheres")
        
        return "; ".join(interpretations)


class IntegratedExplainer:
    """
    Integrated explainer combining GradCAM and symmetry analysis
    Optimized version using only GradCAM for efficiency
    """
    
    def __init__(self, model: nn.Module, target_layer: str, symmetry_analyzer):
        """
        Initialize integrated explainer
        
        Args:
            model: PyTorch model
            target_layer: Target layer for GradCAM
            symmetry_analyzer: Symmetry analyzer instance
        """
        self.gradcam = GradCAM(model, target_layer)
        self.symmetry_explainer = SymmetryExplainer(symmetry_analyzer)
        self.model = model
        
        # Class names
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    def explain_prediction(self, image: torch.Tensor, original_image: np.ndarray, 
                          class_idx: Optional[int] = None) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            image: Preprocessed image tensor
            original_image: Original image for symmetry analysis
            class_idx: Target class index
            
        Returns:
            Dictionary with all explanations
        """
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                if len(self.model.forward.__code__.co_varnames) > 2:  # Has return_features parameter
                    output = self.model(image.unsqueeze(0), return_features=True)
                    if isinstance(output, dict):
                        logits = output['logits']
                        symmetry_features = output.get('symmetry_features', None)
                    else:
                        logits = output[0] if isinstance(output, tuple) else output
                        symmetry_features = None
                else:
                    output = self.model(image.unsqueeze(0))
                    logits = output[0] if isinstance(output, tuple) else output
                    symmetry_features = None
            else:
                logits = self.model(image.unsqueeze(0))
                symmetry_features = None
        
        probabilities = F.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        if class_idx is None:
            class_idx = predicted_class
        
        # Generate GradCAM
        gradcam_heatmap = self.gradcam.generate_cam(image.unsqueeze(0), class_idx)
        
        # Generate symmetry explanations
        symmetry_features_dict = self.symmetry_explainer.symmetry_analyzer.extract_all_symmetry_features(original_image)
        symmetry_explanations = self.symmetry_explainer.explain_symmetry_features(
            original_image, symmetry_features_dict
        )
        
        # Combine explanations
        explanation = {
            'prediction': {
                'predicted_class': predicted_class,
                'predicted_class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'all_probabilities': probabilities[0].cpu().numpy(),
                'class_names': self.class_names
            },
            'visual_explanations': {
                'gradcam': gradcam_heatmap,
                'asymmetry_map': symmetry_explanations['asymmetry_map']
            },
            'symmetry_analysis': {
                'features': symmetry_features_dict,
                'feature_importance': symmetry_explanations['feature_importance'],
                'midline_analysis': symmetry_explanations['midline_analysis'],
                'hemisphere_comparison': symmetry_explanations['hemisphere_comparison'],
                'clinical_interpretation': symmetry_explanations['clinical_interpretation']
            }
        }
        
        return explanation
    
    def visualize_explanation(self, explanation: Dict, original_image: np.ndarray, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of explanations
        
        Args:
            explanation: Explanation dictionary
            original_image: Original input image
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout - optimized to 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # GradCAM
        ax2 = fig.add_subplot(gs[0, 1])
        gradcam_overlay = self._overlay_heatmap(original_image, explanation['visual_explanations']['gradcam'])
        ax2.imshow(gradcam_overlay)
        ax2.set_title('GradCAM Explanation')
        ax2.axis('off')
        
        # Asymmetry map
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(explanation['visual_explanations']['asymmetry_map'], cmap='hot')
        ax3.set_title('Asymmetry Map')
        ax3.axis('off')
        
        # Prediction probabilities
        ax5 = fig.add_subplot(gs[1, :2])
        probs = explanation['prediction']['all_probabilities']
        class_names = explanation['prediction']['class_names']
        bars = ax5.bar(class_names, probs)
        ax5.set_title(f'Prediction: {explanation["prediction"]["predicted_class_name"]} '
                     f'(Confidence: {explanation["prediction"]["confidence"]:.3f})')
        ax5.set_ylabel('Probability')
        ax5.tick_params(axis='x', rotation=45)
        
        # Highlight predicted class
        predicted_idx = explanation['prediction']['predicted_class']
        bars[predicted_idx].set_color('red')
        
        # Symmetry features
        ax6 = fig.add_subplot(gs[1, 2])
        features = explanation['symmetry_analysis']['features']
        feature_names = list(features.keys())
        feature_values = list(features.values())
        ax6.barh(feature_names, feature_values, color='skyblue')
        ax6.set_title('Symmetry Features')
        ax6.set_xlabel('Score')
        ax6.set_xlim(0, 1)
        
        # Clinical interpretation
        ax7 = fig.add_subplot(gs[2, :])
        clinical_text = explanation['symmetry_analysis']['clinical_interpretation']
        ax7.text(0.05, 0.5, clinical_text, transform=ax7.transAxes, fontsize=12,
                verticalalignment='center', wrap=True)
        ax7.set_title('Clinical Interpretation')
        ax7.axis('off')
        
        plt.suptitle('Brain Tumor Classification - Explainable AI Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image
            heatmap: Heatmap to overlay
            alpha: Transparency of heatmap
            
        Returns:
            Overlayed image
        """
        # Normalize image
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_normalized = image
        else:
            img_normalized = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image
        
        if img_normalized.max() <= 1.0:
            img_normalized = (img_normalized * 255).astype(np.uint8)
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img_normalized.shape[1], img_normalized.shape[0]))
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(img_normalized, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def cleanup(self):
        """Clean up hooks"""
        self.gradcam.remove_hooks()


# Utility functions
def create_explainer(model: nn.Module, target_layer: str, symmetry_analyzer) -> IntegratedExplainer:
    """
    Factory function to create integrated explainer
    
    Args:
        model: PyTorch model
        target_layer: Target layer name
        symmetry_analyzer: Symmetry analyzer instance
        
    Returns:
        IntegratedExplainer instance
    """
    return IntegratedExplainer(model, target_layer, symmetry_analyzer)


def batch_explain(explainer: IntegratedExplainer, images: torch.Tensor, 
                 original_images: List[np.ndarray]) -> List[Dict]:
    """
    Generate explanations for a batch of images
    
    Args:
        explainer: IntegratedExplainer instance
        images: Batch of preprocessed images
        original_images: List of original images
        
    Returns:
        List of explanation dictionaries
    """
    explanations = []
    
    for i in range(images.shape[0]):
        explanation = explainer.explain_prediction(images[i], original_images[i])
        explanations.append(explanation)
    
    return explanations


# Example usage
if __name__ == "__main__":
    # This would be used with an actual model and data
    print("GradCAM and Explainable AI module loaded successfully!")
    print("Use with trained model and symmetry analyzer for explanations.")
