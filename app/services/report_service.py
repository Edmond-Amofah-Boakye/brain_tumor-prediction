"""
Report Service
Handles clinical report generation
"""

from datetime import datetime
import numpy as np
from typing import Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import REPORT_CONFIG, CLASS_NAMES


class ReportService:
    """Service for report generation"""
    
    @staticmethod
    def generate_clinical_report(
        prediction_results: Dict,
        symmetry_features: Dict,
        clinical_interpretation: str
    ) -> str:
        """
        Generate comprehensive clinical report
        
        Args:
            prediction_results: Model prediction results
            symmetry_features: Symmetry analysis features
            clinical_interpretation: Clinical interpretation text
            
        Returns:
            Formatted clinical report as markdown
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
## Brain Tumor Classification Report

**Generated:** {timestamp}  
**System:** NeuroScan AI - Symmetry-Integrated CNN  
**Model Accuracy:** 95.82%

---

### Analysis Results

**Primary Diagnosis:** {prediction_results['class_name']}  
**Confidence Level:** {prediction_results['confidence']:.1%}  
**Model Uncertainty:** {prediction_results['uncertainty'][0].item():.3f}

### Class Probabilities

"""
        
        # Add probabilities for all classes
        for i, class_name in enumerate(CLASS_NAMES):
            prob = prediction_results['probabilities'][0][i].item()
            report += f"- **{class_name}:** {prob:.1%}\n"
        
        report += """
---

### Symmetry Analysis (Core 4 Metrics)

"""
        
        # Add symmetry metrics
        for metric_name, value in symmetry_features.items():
            # Format metric name
            display_name = metric_name.replace('_', ' ').title()
            report += f"**{display_name}:** {value:.3f}\n\n"
        
        report += f"""
**Overall Symmetry Score:** {np.mean(list(symmetry_features.values())):.3f}

---

### Clinical Interpretation

{clinical_interpretation}

---

### Recommendations

"""
        
        # Add recommendations based on confidence
        confidence = prediction_results['confidence']
        class_name = prediction_results['class_name']
        
        if confidence > 0.8:
            report += "✅ **High confidence prediction.** Consider correlation with clinical symptoms and patient history.\n\n"
        elif confidence > 0.6:
            report += "⚠️ **Medium confidence prediction.** Recommend additional imaging or expert radiologist review.\n\n"
        else:
            report += "❌ **Low confidence prediction.** Strongly recommend expert radiologist review and consider additional diagnostic tests.\n\n"
        
        if class_name != 'No Tumor':
            report += "- Tumor detected - recommend oncology consultation\n"
            report += "- Consider additional imaging modalities (MRI, CT, PET scan) for treatment planning\n"
            report += "- Initiate appropriate referral pathways based on tumor type\n\n"
        
        report += f"""
---

### Disclaimer

{REPORT_CONFIG['disclaimer']}

---

**Report ID:** {timestamp.replace(':', '-').replace(' ', '_')}  
**System Version:** 1.0
"""
        
        return report
    
    @staticmethod
    def generate_filename(prefix: str = "brain_tumor_report") -> str:
        """
        Generate filename for report
        
        Args:
            prefix: Filename prefix
            
        Returns:
            Formatted filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{prefix}_{timestamp}.md"
