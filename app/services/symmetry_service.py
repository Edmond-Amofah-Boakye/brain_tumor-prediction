"""
Symmetry Service
Handles brain symmetry analysis using the existing BrainSymmetryAnalyzer
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.symmetry_analyzer import BrainSymmetryAnalyzer
from app.config import CLINICAL_THRESHOLDS, CLASS_NAMES


class SymmetryService:
    """Service for symmetry analysis operations"""
    
    def __init__(self):
        """Initialize symmetry analyzer"""
        self.analyzer = BrainSymmetryAnalyzer(
            image_size=(224, 224)
        )
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Perform complete symmetry analysis
        
        Args:
            image: Brain scan image as numpy array
            
        Returns:
            Dictionary containing symmetry features and clinical interpretation
        """
        # Extract core 4 symmetry metrics
        features = self.analyzer.extract_all_symmetry_features(image)
        
        # Get clinical interpretation
        tumor_detected = False  # Will be set by caller
        tumor_type = None
        
        return {
            'features': features,
            'processed_image': self.analyzer.processed_image,
            'midline': self.analyzer.midline,
            'left_hemisphere': self.analyzer.left_hemisphere,
            'right_hemisphere': self.analyzer.right_hemisphere
        }
    
    def generate_clinical_interpretation(
        self, 
        features: Dict[str, float],
        tumor_detected: bool = False,
        tumor_type: Optional[str] = None
    ) -> str:
        """
        Generate clinical interpretation based on symmetry features
        
        Args:
            features: Symmetry feature dictionary
            tumor_detected: Whether CNN detected a tumor
            tumor_type: Type of tumor detected (if any)
            
        Returns:
            Clinical interpretation text
        """
        # Get metrics with backward compatibility
        intensity_bal = features.get('hemisphere_intensity_balance', 
                                    features.get('intensity_symmetry', 0))
        structural_bal = features.get('hemisphere_structural_balance',
                                     features.get('structural_symmetry', 0))
        asymmetry_idx = features.get('hemisphere_asymmetry_index',
                                     features.get('asymmetry_index', 0))
        abnormality = features.get('tissue_abnormality_score', 0)
        
        # Calculate overall balance
        avg_balance = (intensity_bal + structural_bal) / 2
        
        # Get thresholds
        abn_thresholds = CLINICAL_THRESHOLDS['abnormality_score']
        
        # Generate context-aware interpretation
        if tumor_detected:
            if abnormality > abn_thresholds['high']:
                return (
                    f"**{tumor_type} Detected with Significant Tissue Abnormality**\n\n"
                    f"**Abnormal Tissue:** {abnormality*100:.1f}% of brain shows abnormal intensity patterns\n"
                    f"**Hemisphere Balance:** {avg_balance:.2f} "
                    f"({'High - tumor centrally located' if avg_balance > 0.7 else 'Low - tumor causing displacement'})\n\n"
                    f"**Clinical Interpretation:** Substantial pathological changes detected. "
                    f"{'The high hemisphere balance suggests central tumor location (e.g., pituitary, midline glioma).' if avg_balance > 0.7 else 'The low hemisphere balance indicates lateral tumor with mass effect.'}"
                )
            elif abnormality > abn_thresholds['moderate']:
                return (
                    f"**{tumor_type} Detected with Moderate Tissue Abnormality**\n\n"
                    f"**Abnormal Tissue:** {abnormality*100:.1f}% of brain\n"
                    f"**Hemisphere Balance:** {avg_balance:.2f}\n\n"
                    f"**Clinical Interpretation:** Moderate pathological changes detected. "
                    f"Tumor may be in early stage or well-circumscribed."
                )
            else:
                return (
                    f"**{tumor_type} Detected - Low Structural Abnormality Score**\n\n"
                    f"**Abnormal Tissue:** {abnormality*100:.1f}%\n\n"
                    f"**Note:** Low abnormality score despite CNN tumor detection. "
                    f"This may indicate: (1) Very early stage tumor, (2) Tumor with similar intensity to normal tissue, "
                    f"or (3) Need for expert radiologist review to confirm findings."
                )
        else:
            # No tumor case
            if abnormality > abn_thresholds['moderate']:
                return (
                    f"**No Tumor Detected BUT Structural Abnormality Present**\n\n"
                    f"**Abnormal Tissue:** {abnormality*100:.1f}%\n\n"
                    f"⚠️ **Important:** CNN classifies as 'No Tumor' but {abnormality*100:.1f}% abnormal tissue detected. "
                    f"This discrepancy warrants expert radiologist review. Possible causes: artifact, non-tumor pathology, "
                    f"or false negative that requires investigation."
                )
            elif abnormality > abn_thresholds['low']:
                return (
                    f"**No Tumor - Minor Structural Variation**\n\n"
                    f"**Abnormal Tissue:** {abnormality*100:.1f}%\n"
                    f"**Hemisphere Balance:** {avg_balance:.2f}\n\n"
                    f"**Clinical Interpretation:** Minimal structural variation detected. "
                    f"Likely within normal range, but clinical correlation recommended."
                )
            else:
                return (
                    f"**No Tumor - Normal Structural Analysis**\n\n"
                    f"**Abnormal Tissue:** {abnormality*100:.1f}%\n"
                    f"**Hemisphere Balance:** {avg_balance:.2f}\n\n"
                    f"**Clinical Interpretation:** Structural analysis consistent with healthy brain tissue. "
                    f"Both CNN classification and structural metrics indicate normal findings."
                )
    
    def create_asymmetry_map(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Create asymmetry heatmap
        
        Args:
            left: Left hemisphere image
            right: Right hemisphere (already flipped)
            
        Returns:
            Asymmetry map
        """
        if left.shape != right.shape:
            return np.zeros((224, 224))
        
        asymmetry_map = np.abs(left.astype(np.float32) - right.astype(np.float32))
        return asymmetry_map
    
    def get_metric_interpretation(self, metric_name: str, value: float) -> str:
        """
        Get interpretation for a specific metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            Interpretation string
        """
        interpretations = {
            'hemisphere_intensity_balance': {
                'desc': 'Intensity Balance',
                'high': f'High symmetry ({value:.2f}) - hemispheres have similar intensity patterns',
                'low': f'Low symmetry ({value:.2f}) - significant intensity differences detected'
            },
            'hemisphere_structural_balance': {
                'desc': 'Structural Balance', 
                'high': f'High symmetry ({value:.2f}) - similar structural features',
                'low': f'Low symmetry ({value:.2f}) - structural differences present'
            },
            'hemisphere_asymmetry_index': {
                'desc': 'Asymmetry Index',
                'high': f'High asymmetry ({value:.2f}) - significant differences between hemispheres',
                'low': f'Low asymmetry ({value:.2f}) - hemispheres are relatively symmetric'
            },
            'tissue_abnormality_score': {
                'desc': 'Tissue Abnormality',
                'high': f'High abnormality ({value:.2f}) - pathological tissue patterns detected',
                'low': f'Low abnormality ({value:.2f}) - tissue appears normal'
            }
        }
        
        if metric_name not in interpretations:
            return f"{metric_name}: {value:.2f}"
        
        metric_info = interpretations[metric_name]
        
        # Determine if high or low (thresholds vary by metric)
        if 'asymmetry' in metric_name or 'abnormality' in metric_name:
            # For these, high values indicate problems
            if value > 0.5:
                return metric_info['high']
            else:
                return metric_info['low']
        else:
            # For balance metrics, high is good
            if value > 0.7:
                return metric_info['high']
            else:
                return metric_info['low']
    
    def get_analyzer(self) -> BrainSymmetryAnalyzer:
        """Get the underlying symmetry analyzer"""
        return self.analyzer
