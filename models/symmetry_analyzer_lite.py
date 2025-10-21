"""
Optimized Brain Symmetry Analysis Module
Reduced from 8 to 3 most impactful metrics for efficiency
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BrainSymmetryAnalyzerLite:
    """
    Lightweight brain symmetry analyzer with 3 core metrics.
    Optimized for performance while maintaining clinical relevance.
    
    Metrics:
    1. Intensity Symmetry - Direct hemisphere pixel comparison
    2. Structural Symmetry - Edge-based geometric analysis  
    3. Asymmetry Index - Overall asymmetry quantification
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.symmetry_metrics = {}
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess brain image for symmetry analysis
        
        Args:
            image: Input brain image (H, W, C) or (H, W)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = image[:, :, 0]
        
        # Resize to standard size
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 255]
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # Apply Gaussian smoothing to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    def detect_midline(self, image: np.ndarray) -> int:
        """
        Detect brain midline using symmetry-based detection
        
        Args:
            image: Preprocessed brain image
            
        Returns:
            Midline x-coordinate
        """
        height, width = image.shape
        
        # Method 1: Center of mass
        moments = cv2.moments(image)
        if moments['m00'] != 0:
            cx_mass = int(moments['m10'] / moments['m00'])
        else:
            cx_mass = width // 2
        
        # Method 2: Symmetry-based detection
        best_correlation = -1
        best_x = width // 2
        
        # Search around center
        search_range = range(width // 4, 3 * width // 4, 2)  # Step by 2 for speed
        
        for x in search_range:
            left_half = image[:, :x]
            right_half = image[:, x:]
            
            # Flip right half and resize to match left half
            right_flipped = cv2.flip(right_half, 1)
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            
            if min_width > 10:  # Minimum width threshold
                left_resized = cv2.resize(left_half, (min_width, height))
                right_resized = cv2.resize(right_flipped, (min_width, height))
                
                # Calculate correlation
                correlation = np.corrcoef(left_resized.flatten(), right_resized.flatten())[0, 1]
                
                if not np.isnan(correlation) and correlation > best_correlation:
                    best_correlation = correlation
                    best_x = x
        
        # Combine methods (weighted average favoring symmetry-based)
        midline = int(0.3 * cx_mass + 0.7 * best_x)
        
        # Ensure midline is within reasonable bounds
        midline = max(width // 4, min(3 * width // 4, midline))
        
        return midline
    
    def split_hemispheres(self, image: np.ndarray, midline: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split brain image into left and right hemispheres
        
        Args:
            image: Preprocessed brain image
            midline: Midline x-coordinate
            
        Returns:
            Tuple of (left_hemisphere, right_hemisphere_flipped)
        """
        left_hemisphere = image[:, :midline]
        right_hemisphere = image[:, midline:]
        
        # Flip right hemisphere for comparison
        right_hemisphere_flipped = cv2.flip(right_hemisphere, 1)
        
        # Resize to same dimensions
        target_width = min(left_hemisphere.shape[1], right_hemisphere_flipped.shape[1])
        height = image.shape[0]
        
        if target_width > 0:
            left_resized = cv2.resize(left_hemisphere, (target_width, height))
            right_resized = cv2.resize(right_hemisphere_flipped, (target_width, height))
        else:
            left_resized = left_hemisphere
            right_resized = right_hemisphere_flipped
        
        return left_resized, right_resized
    
    def compute_intensity_symmetry(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute intensity-based symmetry metric
        Core metric #1: Direct pixel intensity comparison
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere image (already flipped)
            
        Returns:
            Intensity symmetry score (0-1, higher is more symmetric)
        """
        if left.shape != right.shape:
            return 0.0
        
        # Normalize intensities
        left_norm = left.astype(np.float32) / 255.0
        right_norm = right.astype(np.float32) / 255.0
        
        # Calculate absolute difference
        diff = np.abs(left_norm - right_norm)
        
        # Symmetry score (1 - normalized mean absolute difference)
        symmetry_score = 1.0 - np.mean(diff)
        
        return max(0.0, symmetry_score)
    
    def compute_structural_symmetry(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute structural symmetry using edge and contour analysis
        Core metric #2: Geometric feature comparison
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere image
            
        Returns:
            Structural symmetry score (0-1)
        """
        if left.shape != right.shape:
            return 0.0
        
        # Edge detection
        left_edges = cv2.Canny(left, 50, 150)
        right_edges = cv2.Canny(right, 50, 150)
        
        # Calculate edge density
        left_edge_density = np.sum(left_edges > 0) / left_edges.size
        right_edge_density = np.sum(right_edges > 0) / right_edges.size
        
        # Edge density symmetry
        edge_symmetry = 1.0 - abs(left_edge_density - right_edge_density)
        
        # Structural correlation
        if np.std(left_edges) > 0 and np.std(right_edges) > 0:
            correlation = np.corrcoef(left_edges.flatten(), right_edges.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Combine metrics (weighted)
        structural_symmetry = 0.5 * edge_symmetry + 0.5 * max(0.0, correlation)
        
        return max(0.0, structural_symmetry)
    
    def compute_asymmetry_index(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute overall asymmetry index
        Core metric #3: Statistical asymmetry measure
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere image
            
        Returns:
            Asymmetry index (0-1, lower is more symmetric)
        """
        if left.shape != right.shape:
            return 1.0
        
        # Normalized cross-correlation
        left_norm = (left - np.mean(left)) / (np.std(left) + 1e-8)
        right_norm = (right - np.mean(right)) / (np.std(right) + 1e-8)
        
        # Calculate normalized cross-correlation
        correlation = np.mean(left_norm * right_norm)
        
        # Convert to symmetry score (inverse of asymmetry)
        # Higher correlation = more symmetric = lower asymmetry
        symmetry_score = max(0.0, correlation)
        
        return max(0.0, min(1.0, symmetry_score))
    
    def extract_all_symmetry_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all 3 core symmetry features from brain image
        
        Args:
            image: Input brain image
            
        Returns:
            Dictionary with 3 symmetry features
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Detect midline
        midline = self.detect_midline(processed_image)
        
        # Split hemispheres
        left_hemisphere, right_hemisphere = self.split_hemispheres(processed_image, midline)
        
        # Compute all 3 core metrics
        features = {
            'intensity_symmetry': self.compute_intensity_symmetry(left_hemisphere, right_hemisphere),
            'structural_symmetry': self.compute_structural_symmetry(left_hemisphere, right_hemisphere),
            'asymmetry_index': self.compute_asymmetry_index(left_hemisphere, right_hemisphere)
        }
        
        # Store for visualization
        self.symmetry_metrics = features
        self.processed_image = processed_image
        self.midline = midline
        self.left_hemisphere = left_hemisphere
        self.right_hemisphere = right_hemisphere
        
        return features
    
    def visualize_symmetry_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of symmetry analysis
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, 'processed_image'):
            raise ValueError("Must run extract_all_symmetry_features first")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Brain Symmetry Analysis (Optimized - 3 Metrics)', fontsize=14, fontweight='bold')
        
        # Original image with midline
        axes[0, 0].imshow(self.processed_image, cmap='gray')
        axes[0, 0].axvline(x=self.midline, color='red', linewidth=2, label='Detected Midline')
        axes[0, 0].set_title('Original Image with Midline')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # Left hemisphere
        axes[0, 1].imshow(self.left_hemisphere, cmap='gray')
        axes[0, 1].set_title('Left Hemisphere')
        axes[0, 1].axis('off')
        
        # Difference map
        if self.left_hemisphere.shape == self.right_hemisphere.shape:
            diff_map = np.abs(self.left_hemisphere.astype(np.float32) - 
                            self.right_hemisphere.astype(np.float32))
            axes[1, 0].imshow(diff_map, cmap='hot')
            axes[1, 0].set_title('Asymmetry Map')
            axes[1, 0].axis('off')
        
        # Symmetry metrics bar plot
        metrics_names = list(self.symmetry_metrics.keys())
        metrics_values = list(self.symmetry_metrics.values())
        
        axes[1, 1].barh(metrics_names, metrics_values, color='skyblue')
        axes[1, 1].set_xlabel('Symmetry Score')
        axes[1, 1].set_title('Core Symmetry Metrics')
        axes[1, 1].set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_feature_vector(self, image: np.ndarray) -> np.ndarray:
        """
        Get symmetry features as numpy array for model input
        
        Args:
            image: Input brain image
            
        Returns:
            Feature vector of shape (3,)
        """
        features = self.extract_all_symmetry_features(image)
        return np.array([
            features['intensity_symmetry'],
            features['structural_symmetry'],
            features['asymmetry_index']
        ], dtype=np.float32)


# Backward compatibility alias
BrainSymmetryAnalyzer = BrainSymmetryAnalyzerLite


# Example usage and testing
if __name__ == "__main__":
    # Test the lite symmetry analyzer
    analyzer = BrainSymmetryAnalyzerLite()
    
    # Create a test image (simulated brain scan)
    test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    
    # Extract symmetry features
    features = analyzer.extract_all_symmetry_features(test_image)
    
    print("Optimized Symmetry Features (3 metrics):")
    for feature_name, value in features.items():
        print(f"{feature_name}: {value:.4f}")
    
    print(f"\nFeature vector shape: {analyzer.get_feature_vector(test_image).shape}")
    
    # Create visualization
    fig = analyzer.visualize_symmetry_analysis()
    plt.show()
