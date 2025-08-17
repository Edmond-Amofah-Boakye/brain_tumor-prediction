"""
Brain Symmetry Analysis Module
Implements advanced symmetry metrics for brain tumor detection
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BrainSymmetryAnalyzer:
    """
    Advanced brain symmetry analysis for tumor detection.
    Computes multiple symmetry metrics that are clinically relevant.
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
        Detect brain midline using multiple methods
        
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
        symmetry_scores = []
        for x in range(width // 4, 3 * width // 4):
            left_half = image[:, :x]
            right_half = image[:, x:]
            
            # Flip right half and resize to match left half
            right_flipped = cv2.flip(right_half, 1)
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            
            if min_width > 0:
                left_resized = cv2.resize(left_half, (min_width, height))
                right_resized = cv2.resize(right_flipped, (min_width, height))
                
                # Calculate correlation
                correlation = cv2.matchTemplate(left_resized, right_resized, cv2.TM_CCOEFF_NORMED)[0, 0]
                symmetry_scores.append(correlation)
            else:
                symmetry_scores.append(0)
        
        if symmetry_scores:
            best_x = np.argmax(symmetry_scores) + width // 4
        else:
            best_x = width // 2
        
        # Combine methods (weighted average)
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
            Tuple of (left_hemisphere, right_hemisphere)
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
    
    def compute_texture_symmetry(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute texture-based symmetry using GLCM features
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere image
            
        Returns:
            Texture symmetry score (0-1)
        """
        if left.shape != right.shape or left.size == 0:
            return 0.0
        
        try:
            # Compute GLCM features for both hemispheres
            distances = [1, 2]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            # Left hemisphere GLCM
            glcm_left = graycomatrix(left, distances=distances, angles=angles, 
                                   levels=256, symmetric=True, normed=True)
            contrast_left = graycoprops(glcm_left, 'contrast').mean()
            energy_left = graycoprops(glcm_left, 'energy').mean()
            homogeneity_left = graycoprops(glcm_left, 'homogeneity').mean()
            
            # Right hemisphere GLCM
            glcm_right = graycomatrix(right, distances=distances, angles=angles,
                                    levels=256, symmetric=True, normed=True)
            contrast_right = graycoprops(glcm_right, 'contrast').mean()
            energy_right = graycoprops(glcm_right, 'energy').mean()
            homogeneity_right = graycoprops(glcm_right, 'homogeneity').mean()
            
            # Calculate feature differences
            contrast_diff = abs(contrast_left - contrast_right)
            energy_diff = abs(energy_left - energy_right)
            homogeneity_diff = abs(homogeneity_left - homogeneity_right)
            
            # Normalize and combine (lower difference = higher symmetry)
            max_contrast = max(contrast_left, contrast_right, 1e-6)
            max_energy = max(energy_left, energy_right, 1e-6)
            max_homogeneity = max(homogeneity_left, homogeneity_right, 1e-6)
            
            texture_symmetry = 1.0 - (
                (contrast_diff / max_contrast) +
                (energy_diff / max_energy) +
                (homogeneity_diff / max_homogeneity)
            ) / 3.0
            
            return max(0.0, texture_symmetry)
            
        except Exception:
            return 0.0
    
    def compute_structural_symmetry(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute structural symmetry using edge and contour analysis
        
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
        
        # Combine metrics
        structural_symmetry = 0.5 * edge_symmetry + 0.5 * max(0.0, correlation)
        
        return max(0.0, structural_symmetry)
    
    def compute_statistical_symmetry(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute statistical symmetry using moment-based features
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere image
            
        Returns:
            Statistical symmetry score (0-1)
        """
        if left.shape != right.shape:
            return 0.0
        
        # Calculate statistical moments
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        left_std = np.std(left)
        right_std = np.std(right)
        left_skew = self._calculate_skewness(left)
        right_skew = self._calculate_skewness(right)
        
        # Calculate symmetry for each statistic
        mean_symmetry = 1.0 - abs(left_mean - right_mean) / 255.0
        std_symmetry = 1.0 - abs(left_std - right_std) / 255.0
        skew_symmetry = 1.0 - abs(left_skew - right_skew) / 2.0  # Skewness typically in [-2, 2]
        
        # Combine statistics
        statistical_symmetry = (mean_symmetry + std_symmetry + skew_symmetry) / 3.0
        
        return max(0.0, statistical_symmetry)
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensity distribution"""
        flat = image.flatten().astype(np.float32)
        mean_val = np.mean(flat)
        std_val = np.std(flat)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((flat - mean_val) / std_val) ** 3)
        return skewness
    
    def compute_volume_asymmetry(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        Compute volume asymmetry ratio
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere image
            
        Returns:
            Volume asymmetry score (0-1, higher is more symmetric)
        """
        # Threshold images to get brain tissue
        left_thresh = left > np.mean(left)
        right_thresh = right > np.mean(right)
        
        # Calculate volumes (number of pixels above threshold)
        left_volume = np.sum(left_thresh)
        right_volume = np.sum(right_thresh)
        
        if left_volume + right_volume == 0:
            return 0.0
        
        # Volume asymmetry ratio
        total_volume = left_volume + right_volume
        volume_diff = abs(left_volume - right_volume)
        volume_symmetry = 1.0 - (volume_diff / total_volume)
        
        return max(0.0, volume_symmetry)
    
    def extract_all_symmetry_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all symmetry features from brain image
        
        Args:
            image: Input brain image
            
        Returns:
            Dictionary of symmetry features
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Detect midline
        midline = self.detect_midline(processed_image)
        
        # Split hemispheres
        left_hemisphere, right_hemisphere = self.split_hemispheres(processed_image, midline)
        
        # Compute all symmetry metrics
        features = {
            'intensity_symmetry': self.compute_intensity_symmetry(left_hemisphere, right_hemisphere),
            'texture_symmetry': self.compute_texture_symmetry(left_hemisphere, right_hemisphere),
            'structural_symmetry': self.compute_structural_symmetry(left_hemisphere, right_hemisphere),
            'statistical_symmetry': self.compute_statistical_symmetry(left_hemisphere, right_hemisphere),
            'volume_asymmetry': self.compute_volume_asymmetry(left_hemisphere, right_hemisphere),
            'midline_position': midline / processed_image.shape[1],  # Normalized midline position
            'hemisphere_correlation': self._compute_correlation(left_hemisphere, right_hemisphere),
            'asymmetry_index': self._compute_asymmetry_index(left_hemisphere, right_hemisphere)
        }
        
        # Store for visualization
        self.symmetry_metrics = features
        self.processed_image = processed_image
        self.midline = midline
        self.left_hemisphere = left_hemisphere
        self.right_hemisphere = right_hemisphere
        
        return features
    
    def _compute_correlation(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute correlation between hemispheres"""
        if left.shape != right.shape:
            return 0.0
        
        left_flat = left.flatten().astype(np.float32)
        right_flat = right.flatten().astype(np.float32)
        
        if np.std(left_flat) == 0 or np.std(right_flat) == 0:
            return 0.0
        
        correlation = np.corrcoef(left_flat, right_flat)[0, 1]
        return 0.0 if np.isnan(correlation) else max(0.0, correlation)
    
    def _compute_asymmetry_index(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute overall asymmetry index"""
        if left.shape != right.shape:
            return 1.0
        
        # Normalized cross-correlation
        left_norm = (left - np.mean(left)) / (np.std(left) + 1e-8)
        right_norm = (right - np.mean(right)) / (np.std(right) + 1e-8)
        
        # Calculate normalized cross-correlation
        correlation = np.mean(left_norm * right_norm)
        asymmetry_index = 1.0 - max(0.0, correlation)
        
        return min(1.0, max(0.0, asymmetry_index))
    
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
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Brain Symmetry Analysis', fontsize=16, fontweight='bold')
        
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
        
        # Right hemisphere (flipped)
        axes[0, 2].imshow(self.right_hemisphere, cmap='gray')
        axes[0, 2].set_title('Right Hemisphere (Flipped)')
        axes[0, 2].axis('off')
        
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
        axes[1, 1].set_title('Symmetry Metrics')
        axes[1, 1].set_xlim(0, 1)
        
        # Overall symmetry score
        overall_score = np.mean(list(self.symmetry_metrics.values()))
        axes[1, 2].pie([overall_score, 1-overall_score], 
                      labels=['Symmetric', 'Asymmetric'],
                      colors=['lightgreen', 'lightcoral'],
                      autopct='%1.1f%%')
        axes[1, 2].set_title(f'Overall Symmetry: {overall_score:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Test the symmetry analyzer
    analyzer = BrainSymmetryAnalyzer()
    
    # Create a test image (simulated brain scan)
    test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    
    # Extract symmetry features
    features = analyzer.extract_all_symmetry_features(test_image)
    
    print("Symmetry Features:")
    for feature_name, value in features.items():
        print(f"{feature_name}: {value:.4f}")
    
    # Create visualization
    fig = analyzer.visualize_symmetry_analysis()
    plt.show()
