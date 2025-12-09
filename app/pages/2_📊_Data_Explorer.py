"""
Data Exploration Page
Visualize dataset statistics, class distributions, and preprocessing effects
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from PIL import Image
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.config import DATA_DIR, CLASS_NAMES, PREPROCESSING_CONFIG


def load_dataset_info():
    """Load dataset information"""
    # Try to find dataset directory
    dataset_paths = [
        DATA_DIR / "raw" / "dataset",
        DATA_DIR / "raw",
        Path("data/raw/dataset"),
    ]
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            return dataset_path
    
    return None


def count_images_by_class(dataset_path, split_name="Training"):
    """Count images in each class"""
    class_counts = {}
    
    split_path = dataset_path / split_name
    if not split_path.exists():
        return class_counts
    
    # Map folder names to class names
    folder_map = {
        'glioma_tumor': 'Glioma Tumor',
        'glioma': 'Glioma Tumor',
        'meningioma_tumor': 'Meningioma Tumor',
        'meningioma': 'Meningioma Tumor',
        'no_tumor': 'No Tumor',
        'notumor': 'No Tumor',
        'pituitary_tumor': 'Pituitary Tumor',
        'pituitary': 'Pituitary Tumor'
    }
    
    for folder in split_path.iterdir():
        if folder.is_dir():
            folder_name = folder.name.lower()
            class_name = folder_map.get(folder_name, folder.name)
            
            # Count image files
            image_files = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + \
                         list(folder.glob('*.jpeg')) + list(folder.glob('*.bmp'))
            
            if class_name in class_counts:
                class_counts[class_name] += len(image_files)
            else:
                class_counts[class_name] = len(image_files)
    
    return class_counts


def render_dataset_overview(dataset_path):
    """Render dataset overview section"""
    st.header("📊 Dataset Overview")
    
    # Get counts for each split
    training_counts = count_images_by_class(dataset_path, "Training")
    testing_counts = count_images_by_class(dataset_path, "Testing")
    
    if not training_counts and not testing_counts:
        st.warning("⚠️ No dataset found. Please ensure your dataset is in the correct location.")
        st.info(f"Expected location: {dataset_path}")
        return
    
    # Calculate totals
    total_train = sum(training_counts.values())
    total_test = sum(testing_counts.values())
    total_all = total_train + total_test
    
    # Display total statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", f"{total_all:,}")
    with col2:
        st.metric("Training", f"{total_train:,}")
    with col3:
        st.metric("Testing", f"{total_test:,}")
    with col4:
        st.metric("Classes", len(CLASS_NAMES))
    
    st.divider()
    
    # Create class distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Set Distribution")
        if training_counts:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(training_counts.keys()),
                    y=list(training_counts.values()),
                    marker_color='skyblue',
                    text=list(training_counts.values()),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                xaxis_title="Class",
                yaxis_title="Number of Images",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentages
            st.markdown("**Class Distribution:**")
            for class_name, count in training_counts.items():
                percentage = (count / total_train) * 100
                st.write(f"- {class_name}: {count} ({percentage:.1f}%)")
    
    with col2:
        st.subheader("Testing Set Distribution")
        if testing_counts:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(testing_counts.keys()),
                    y=list(testing_counts.values()),
                    marker_color='lightcoral',
                    text=list(testing_counts.values()),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                xaxis_title="Class",
                yaxis_title="Number of Images",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentages
            st.markdown("**Class Distribution:**")
            for class_name, count in testing_counts.items():
                percentage = (count / total_test) * 100
                st.write(f"- {class_name}: {count} ({percentage:.1f}%)")
    
    # Combined pie chart
    st.subheader("Overall Class Balance")
    all_counts = {}
    for class_name in CLASS_NAMES:
        train_c = training_counts.get(class_name, 0)
        test_c = testing_counts.get(class_name, 0)
        all_counts[class_name] = train_c + test_c
    
    fig = go.Figure(data=[go.Pie(
        labels=list(all_counts.keys()),
        values=list(all_counts.values()),
        hole=0.3
    )])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_preprocessing_info():
    """Render preprocessing information"""
    st.header("🔧 Preprocessing Pipeline")
    
    st.markdown("""
    ### Image Preprocessing Steps
    
    The following preprocessing steps are applied to all images before model input:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Image Loading**
        - Load image using PIL
        - Convert to RGB (3 channels)
        
        **2. Resizing**
        - Target size: 224×224 pixels
        - Maintains aspect ratio
        - Bilinear interpolation
        
        **3. Normalization**
        - Pixel values: [0, 255] → [0, 1]
        - Divide by 255.0
        """)
    
    with col2:
        st.markdown(f"""
        **4. ImageNet Normalization**
        - Mean: {PREPROCESSING_CONFIG['mean']}
        - Std: {PREPROCESSING_CONFIG['std']}
        - Formula: (pixel - mean) / std
        
        **5. Tensor Conversion**
        - Convert to PyTorch tensor
        - Shape: (1, 3, 224, 224)
        - Data type: float32
        """)
    
    st.info("""
    💡 **Why ImageNet Normalization?**
    
    We use ImageNet normalization because our CNN backbone (EfficientNet-B3) 
    was pre-trained on ImageNet. This ensures the input distribution matches 
    what the model expects, leading to better performance.
    """)


def render_data_augmentation():
    """Render data augmentation information"""
    st.header("🔄 Data Augmentation (Training Only)")
    
    st.markdown("""
    During training, the following augmentations are applied to increase dataset diversity:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Geometric Transforms**
        - Random rotation (±15°)
        - Random horizontal flip (50%)
        - Random affine transforms
        """)
    
    with col2:
        st.markdown("""
        **Color Transforms**
        - Random brightness (±20%)
        - Random contrast (±20%)
        - Color jitter
        """)
    
    with col3:
        st.markdown("""
        **Quality Transforms**
        - Gaussian blur (occasional)
        - Random noise addition
        - JPEG compression artifacts
        """)
    
    st.warning("""
    ⚠️ **Important:** Augmentations are ONLY applied during training. 
    Test/validation images use standard preprocessing only.
    """)


def render_symmetry_insights():
    """Render insights about symmetry metrics"""
    st.header("🧠 Symmetry Metrics Insights")
    
    st.markdown("""
    ### Core 4 Symmetry Metrics Distribution
    
    Understanding how symmetry metrics vary across different tumor types:
    """)
    
    # Create example data (you would load real statistics from your training)
    st.info("""
    💡 **Expected Patterns:**
    
    - **Glioma Tumors**: Typically show moderate asymmetry (0.4-0.6) due to infiltrative growth
    - **Meningioma Tumors**: Often show high abnormality scores but maintained symmetry
    - **Pituitary Tumors**: Usually central location → high hemisphere balance (>0.7)
    - **No Tumor**: High balance metrics (>0.7), low abnormality (<0.05)
    """)
    
    st.markdown("""
    ### Clinical Thresholds
    
    These thresholds guide clinical interpretation:
    """)
    
    threshold_data = {
        'Metric': [
            'Hemisphere Intensity Balance',
            'Hemisphere Structural Balance',
            'Hemisphere Asymmetry Index',
            'Tissue Abnormality Score'
        ],
        'Normal Range': ['> 0.70', '> 0.65', '< 0.30', '< 0.05'],
        'Abnormal Range': ['< 0.50', '< 0.45', '> 0.50', '> 0.15'],
        'Clinical Significance': [
            'Indicates tissue density differences',
            'Reflects structural deformation',
            'Overall symmetry deviation',
            'Abnormal tissue detection'
        ]
    }
    
    df = pd.DataFrame(threshold_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main():
    """Main function for data explorer page"""
    st.set_page_config(
        page_title="Data Explorer - NeuroScan AI",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Dataset Analysis & Exploration")
    st.markdown("""
    Comprehensive analysis of the brain tumor dataset, preprocessing pipeline, 
    and symmetry metrics insights.
    """)
    
    st.divider()
    
    # Try to load dataset
    dataset_path = load_dataset_info()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset Overview",
        "🔧 Preprocessing",
        "🔄 Augmentation",
        "🧠 Symmetry Insights"
    ])
    
    with tab1:
        if dataset_path:
            render_dataset_overview(dataset_path)
        else:
            st.warning("⚠️ Dataset not found. Please ensure dataset is in the correct location.")
            st.info("""
            Expected structure:
            ```
            data/raw/dataset/
            ├── Training/
            │   ├── glioma_tumor/
            │   ├── meningioma_tumor/
            │   ├── no_tumor/
            │   └── pituitary_tumor/
            └── Testing/
                ├── glioma_tumor/
                ├── meningioma_tumor/
                ├── no_tumor/
                └── pituitary_tumor/
            ```
            """)
    
    with tab2:
        render_preprocessing_info()
    
    with tab3:
        render_data_augmentation()
    
    with tab4:
        render_symmetry_insights()


if __name__ == "__main__":
    main()
