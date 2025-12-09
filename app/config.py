"""
Application Configuration
Centralized configuration for the brain tumor analysis application
"""

from pathlib import Path
from typing import Dict, List

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Model Configuration
MODEL_CONFIG = {
    "num_classes": 4,
    "backbone": "efficientnet_b3",
    "image_size": (224, 224),
    "input_channels": 3,
}

# Class Names and Descriptions
CLASS_NAMES = [
    "Glioma Tumor",
    "Meningioma Tumor", 
    "No Tumor",
    "Pituitary Tumor"
]

CLASS_DESCRIPTIONS = {
    "Glioma Tumor": "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glia cells that surround nerve cells.",
    "Meningioma Tumor": "A tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most are benign.",
    "No Tumor": "Normal brain tissue with no detectable tumor present. Shows healthy brain structure and symmetry.",
    "Pituitary Tumor": "A growth of abnormal cells in the tissues of the pituitary gland, typically located at the base of the brain."
}

# Clinical Thresholds (for symmetry metrics)
CLINICAL_THRESHOLDS = {
    "intensity_balance": {
        "normal": 0.70,
        "abnormal": 0.50
    },
    "structural_balance": {
        "normal": 0.65,
        "abnormal": 0.45
    },
    "asymmetry_index": {
        "normal": 0.30,
        "abnormal": 0.50
    },
    "abnormality_score": {
        "low": 0.05,
        "moderate": 0.10,
        "high": 0.15
    },
    "midline_shift": {
        "normal_mm": 5.0,
        "significant_mm": 10.0
    }
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
    "resize": (224, 224),
}

# Visualization Configuration
VIZ_CONFIG = {
    "figure_dpi": 150,
    "plot_style": "seaborn-v0_8-darkgrid",
    "color_palette": "viridis",
    "primary_color": "#1f77b4",
    "success_color": "#28a745",
    "warning_color": "#ffc107",
    "danger_color": "#dc3545",
}

# Application UI Configuration
UI_CONFIG = {
    "page_title": "NeuroScan AI - Clinical Decision Support",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "app_name": "NeuroScan AI",
    "app_version": "1.0",
    "app_tagline": "Clinical Decision Support System for Brain Tumor Detection",
    "model_accuracy": "95.82%",
}

# GradCAM Configuration
GRADCAM_CONFIG = {
    "target_layer": "backbone.features.7",  # EfficientNet layer
    "colormap": "jet",
    "alpha": 0.7,
}

# Report Configuration
REPORT_CONFIG = {
    "format": "markdown",
    "include_timestamp": True,
    "include_metrics": True,
    "include_visualization": False,  # For file size
    "disclaimer": "⚠️ **IMPORTANT:** This AI system is designed to assist healthcare professionals and should not replace clinical judgment. All results must be reviewed by qualified medical personnel before making treatment decisions."
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "app.log",
}

def get_model_path(model_name: str = "best_model.pth") -> Path:
    """
    Get the path to a trained model file
    
    Args:
        model_name: Name of the model file
        
    Returns:
        Path to the model file
    """
    # Try to find the latest training run
    results_dirs = list(RESULTS_DIR.glob("training_run_*"))
    if results_dirs:
        latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
        model_path = latest_dir / "checkpoints" / model_name
        if model_path.exists():
            return model_path
    
    return None

def get_config_value(section: str, key: str, default=None):
    """
    Get a configuration value
    
    Args:
        section: Configuration section name
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    configs = {
        "model": MODEL_CONFIG,
        "preprocessing": PREPROCESSING_CONFIG,
        "viz": VIZ_CONFIG,
        "ui": UI_CONFIG,
        "gradcam": GRADCAM_CONFIG,
        "report": REPORT_CONFIG,
        "thresholds": CLINICAL_THRESHOLDS,
    }
    
    config = configs.get(section, {})
    return config.get(key, default)
