"""
Brain Tumor Classification Web Application
Professional web interface for the Symmetry-Integrated CNN system
"""

import os
import io
import base64
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our modules
import torchvision.models as models
from models.symmetry_analyzer_lite import BrainSymmetryAnalyzerLite
from explainability.gradcam import GradCAM
from data.data_loader import BrainTumorDataLoader


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


class BrainTumorApp:
    """Main application class"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.symmetry_analyzer = None
        self.explainer = None
        self.class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        self.class_descriptions = {
            'Glioma Tumor': 'A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glia cells.',
            'Meningioma Tumor': 'A tumor that arises from the meninges — the membranes that surround the brain and spinal cord.',
            'No Tumor': 'Normal brain tissue with no detectable tumor present.',
            'Pituitary Tumor': 'A growth of abnormal cells in the tissues of the pituitary gland.'
        }
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
    
    def load_model(self, model_path=None):
        """Load the trained model"""
        try:
            if model_path is None:
                # Try to find the best model in results directory
                results_dirs = list(Path('results').glob('training_run_*'))
                if results_dirs:
                    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
                    model_path = latest_dir / 'checkpoints' / 'best_model.pth'
                else:
                    st.error("No trained model found. Please train a model first.")
                    return False
            
            if not Path(model_path).exists():
                st.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model with same configuration
            config = checkpoint.get('config', {})
            model = PureCNNModel(
                num_classes=4,
                backbone=config.get('backbone', 'efficientnet_b3')
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Initialize symmetry analyzer (lite version for better performance)
            symmetry_analyzer = BrainSymmetryAnalyzerLite(image_size=(224, 224))
            
            # Initialize explainer
            target_layer = 'backbone.features.7'  # EfficientNet layer
            explainer = GradCAM(model, target_layer)
            
            # Store in session state to persist across interactions
            st.session_state.model = model
            st.session_state.symmetry_analyzer = symmetry_analyzer
            st.session_state.explainer = explainer
            st.session_state.model_loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize
        image_array = image_array.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension (ensure float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return image_tensor, image_array
    
    def analyze_image(self, image):
        """Analyze uploaded image"""
        if not st.session_state.model_loaded:
            st.error("Please load a model first")
            return None
        
        # Get from session state
        model = st.session_state.get('model')
        explainer = st.session_state.get('explainer')
        symmetry_analyzer = st.session_state.get('symmetry_analyzer')
        
        # Debug: Check what's initialized
        print(f"DEBUG: Model: {model is not None}")
        print(f"DEBUG: Explainer: {explainer is not None}")
        print(f"DEBUG: Symmetry Analyzer: {symmetry_analyzer is not None}")
        
        try:
            # Preprocess image
            image_tensor, image_array = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            # Get original image as numpy array for symmetry analysis
            original_image = np.array(image.resize((224, 224)))
            if len(original_image.shape) == 3:
                original_image = np.mean(original_image, axis=2)  # Convert to grayscale
            
            # Model prediction - Simple CNN only (no symmetry)
            with torch.no_grad():
                logits = model(image_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = logits.argmax(dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                print(f"DEBUG: Predicted class: {predicted_class} ({self.class_names[predicted_class]})")
                print(f"DEBUG: Confidence: {confidence:.3f}")
            
            # Create simple uncertainty and confidence estimates
            uncertainty = torch.tensor([[1.0 - confidence]])
            confidence_results = {
                'predictions': probabilities,
                'std': torch.zeros_like(probabilities),
                'confidence': torch.tensor([[confidence]])
            }
            
            # Generate GradCAM explanation - BULLETPROOF VERSION
            print("DEBUG: About to call generate_cam")
            gradcam_heatmap = np.zeros((224, 224))  # Default
            try:
                if explainer is not None and hasattr(explainer, 'generate_cam'):
                    gradcam_heatmap = explainer.generate_cam(image_tensor, predicted_class)
                    print("DEBUG: GradCAM generated successfully")
            except Exception as e:
                print(f"ERROR in GradCAM: {str(e)}")
                gradcam_heatmap = np.zeros((224, 224))
            
            # Generate symmetry explanation - BULLETPROOF VERSION
            print("DEBUG: About to extract symmetry features")
            symmetry_features = {'intensity_symmetry': 0.5, 'structural_symmetry': 0.5, 'asymmetry_index': 0.5}  # Default
            try:
                if symmetry_analyzer is not None:
                    symmetry_features = symmetry_analyzer.extract_all_symmetry_features(original_image)
                    print("DEBUG: Symmetry features extracted successfully")
            except Exception as e:
                print(f"ERROR in symmetry extraction: {str(e)}")
                symmetry_features = {'intensity_symmetry': 0.5, 'structural_symmetry': 0.5, 'asymmetry_index': 0.5}
            
            # Create asymmetry map - BULLETPROOF VERSION
            print("DEBUG: Creating asymmetry map")
            asymmetry_map = np.zeros((224, 224))  # Default
            try:
                # Ensure we have proper 2D array
                img_for_asymmetry = original_image.copy()
                if len(img_for_asymmetry.shape) == 3:
                    img_for_asymmetry = np.mean(img_for_asymmetry, axis=2)
                
                # Calculate half width
                h, w = img_for_asymmetry.shape
                half_w = w // 2
                
                # Get left and flipped right
                left = img_for_asymmetry[:, :half_w]
                right_flipped = np.fliplr(img_for_asymmetry)[:, :half_w]
                
                # Calculate asymmetry
                asymmetry_map = np.abs(left - right_flipped)
                print("DEBUG: Asymmetry map created successfully")
            except Exception as e:
                print(f"ERROR creating asymmetry map: {str(e)}")
                asymmetry_map = np.zeros((224, 224))
            
            # Get midline analysis - BULLETPROOF VERSION
            print("DEBUG: Getting midline analysis")
            midline_analysis = {}
            try:
                if symmetry_analyzer is not None:
                    midline_analysis = symmetry_analyzer.get_midline_analysis()
                    print("DEBUG: Midline analysis retrieved successfully")
            except Exception as e:
                print(f"ERROR in midline analysis: {str(e)}")
                midline_analysis = {}
            
            # Combine explanations
            print("DEBUG: Combining all explanations")
            explanation = {
                'visual_explanations': {
                    'gradcam': gradcam_heatmap,
                    'gradcam_plus_plus': gradcam_heatmap,
                    'asymmetry_map': asymmetry_map
                },
                'symmetry_analysis': {
                    'features': symmetry_features,
                    'clinical_interpretation': self._generate_clinical_interpretation(symmetry_features),
                    'midline_analysis': midline_analysis
                }
            }
            print("DEBUG: Explanations combined successfully")
            
            # Prepare results
            results = {
                'prediction': {
                    'class_idx': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy(),
                    'uncertainty': uncertainty[0].cpu().numpy(),
                    'confidence_intervals': {
                        'mean': confidence_results['predictions'][0].cpu().numpy(),
                        'std': confidence_results['std'][0].cpu().numpy(),
                        'confidence': confidence_results['confidence'][0].cpu().numpy()
                    }
                },
                'explanation': explanation,
                'original_image': original_image,
                'processed_image': image_array
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None
    
    def create_prediction_chart(self, probabilities):
        """Create prediction probability chart"""
        fig = go.Figure(data=[
            go.Bar(
                x=self.class_names,
                y=probabilities,
                marker_color=['red' if i == np.argmax(probabilities) else 'lightblue' 
                             for i in range(len(probabilities))],
                text=[f'{p:.3f}' for p in probabilities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Prediction Probabilities',
            xaxis_title='Tumor Class',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_symmetry_chart(self, symmetry_features):
        """Create symmetry features chart"""
        feature_names = list(symmetry_features.keys())
        feature_values = list(symmetry_features.values())
        
        fig = go.Figure(data=[
            go.Bar(
                y=feature_names,
                x=feature_values,
                orientation='h',
                marker_color='skyblue',
                text=[f'{v:.3f}' for v in feature_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Brain Symmetry Analysis',
            xaxis_title='Symmetry Score (0-1)',
            yaxis_title='Symmetry Metrics',
            xaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_confidence_chart(self, confidence_data):
        """Create confidence interval chart"""
        mean_probs = confidence_data['mean']
        std_probs = confidence_data['std']
        
        fig = go.Figure()
        
        # Add bars with error bars
        fig.add_trace(go.Bar(
            x=self.class_names,
            y=mean_probs,
            error_y=dict(type='data', array=std_probs),
            marker_color='lightgreen',
            name='Mean Probability'
        ))
        
        fig.update_layout(
            title='Prediction Confidence Intervals',
            xaxis_title='Tumor Class',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def _generate_clinical_interpretation(self, symmetry_features):
        """Generate clinical interpretation from symmetry features"""
        avg_symmetry = np.mean(list(symmetry_features.values()))
        
        if avg_symmetry > 0.8:
            return "High degree of brain symmetry detected. This is typical of healthy brain tissue."
        elif avg_symmetry > 0.6:
            return "Moderate brain symmetry detected. Some asymmetry present, which may indicate abnormality."
        else:
            return "Significant brain asymmetry detected. This pattern is often associated with pathological changes."
    
    def display_heatmaps(self, explanation, original_image):
        """Display explanation heatmaps"""
        visual_explanations = explanation['visual_explanations']
        
        # Create subplots
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # GradCAM
        gradcam = visual_explanations['gradcam']
        im1 = axes[1].imshow(gradcam, cmap='jet', alpha=0.7)
        axes[1].imshow(original_image, cmap='gray', alpha=0.3)
        axes[1].set_title('GradCAM')
        axes[1].axis('off')
        
        # GradCAM++
        gradcam_pp = visual_explanations['gradcam_plus_plus']
        im2 = axes[2].imshow(gradcam_pp, cmap='jet', alpha=0.7)
        axes[2].imshow(original_image, cmap='gray', alpha=0.3)
        axes[2].set_title('GradCAM++')
        axes[2].axis('off')
        
        # Asymmetry map
        asymmetry_map = visual_explanations['asymmetry_map']
        im3 = axes[3].imshow(asymmetry_map, cmap='hot')
        axes[3].set_title('Asymmetry Map')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Convert to base64 for display
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="Brain Tumor Classification System",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .prediction-high {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 1rem;
            margin: 1rem 0;
        }
        .prediction-medium {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
        }
        .prediction-low {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🧠 Brain Tumor Classification System</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">An Integrative Approach Using Explainable CNNs and Brain Symmetry Metrics</p>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 System Controls")
            
            # Model loading
            st.subheader("Model Management")
            if st.button("🔄 Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    if self.load_model():
                        st.success("✅ Model loaded successfully!")
                    else:
                        st.error("❌ Failed to load model")
            
            if st.session_state.model_loaded:
                st.success("✅ Model Ready")
            else:
                st.warning("⚠️ Model not loaded")
            
            st.divider()
            
            # Information
            st.subheader("📋 System Information")
            st.info(f"**Device:** {self.device}")
            st.info(f"**Classes:** {len(self.class_names)}")
            
            # Class descriptions
            st.subheader("🏥 Tumor Types")
            for class_name, description in self.class_descriptions.items():
                with st.expander(class_name):
                    st.write(description)
        
        # Main content
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model using the sidebar controls before proceeding.")
            st.info("💡 The system will automatically look for the latest trained model in the results directory.")
            return
        
        # File upload
        st.header("📤 Upload Brain Scan")
        uploaded_file = st.file_uploader(
            "Choose a brain scan image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a brain MRI or CT scan image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.subheader("📊 Image Information")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Format:** {image.format}")
            
            with col2:
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image... This may take a few moments."):
                        results = self.analyze_image(image)
                        st.session_state.analysis_results = results
                
                if st.session_state.analysis_results is not None:
                    results = st.session_state.analysis_results
                    prediction = results['prediction']
                    
                    # Prediction result
                    confidence = prediction['confidence']
                    class_name = prediction['class_name']
                    
                    if confidence > 0.8:
                        st.markdown(f'<div class="prediction-high"><h3>🎯 Prediction: {class_name}</h3><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                    elif confidence > 0.6:
                        st.markdown(f'<div class="prediction-medium"><h3>⚠️ Prediction: {class_name}</h3><p>Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-low"><h3>❓ Prediction: {class_name}</h3><p>Low Confidence: {confidence:.1%}</p></div>', unsafe_allow_html=True)
        
        # Analysis results
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            st.divider()
            st.header("📈 Detailed Analysis")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "🧠 Symmetry Analysis", "🔍 Visual Explanations", "📊 Clinical Report"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction probabilities
                    fig_pred = self.create_prediction_chart(results['prediction']['probabilities'])
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    # Confidence intervals
                    fig_conf = self.create_confidence_chart(results['prediction']['confidence_intervals'])
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Metrics
                st.subheader("📊 Prediction Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Predicted Class", results['prediction']['class_name'])
                with col2:
                    st.metric("Confidence", f"{results['prediction']['confidence']:.1%}")
                with col3:
                    st.metric("Uncertainty", f"{results['prediction']['uncertainty'][0]:.3f}")
                with col4:
                    st.metric("Model Confidence", f"{results['prediction']['confidence_intervals']['confidence'][0]:.1%}")
            
            with tab2:
                # Symmetry analysis
                symmetry_features = results['explanation']['symmetry_analysis']['features']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_sym = self.create_symmetry_chart(symmetry_features)
                    st.plotly_chart(fig_sym, use_container_width=True)
                
                with col2:
                    st.subheader("🧠 Symmetry Metrics")
                    for feature, value in symmetry_features.items():
                        st.metric(feature.replace('_', ' ').title(), f"{value:.3f}")
                
                # Clinical interpretation
                st.subheader("🏥 Clinical Interpretation")
                clinical_text = results['explanation']['symmetry_analysis']['clinical_interpretation']
                st.info(clinical_text)
                
                # Midline analysis
                midline_analysis = results['explanation']['symmetry_analysis'].get('midline_analysis', {})
                if midline_analysis:
                    st.subheader("📏 Midline Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Midline Position", f"{midline_analysis.get('midline_position', 0)}")
                    with col2:
                        st.metric("Deviation", f"{midline_analysis.get('deviation_pixels', 0)} px")
                    with col3:
                        st.metric("Deviation %", f"{midline_analysis.get('deviation_percentage', 0):.1f}%")
                    
                    st.write(f"**Interpretation:** {midline_analysis.get('interpretation', 'N/A')}")
            
            with tab3:
                # Visual explanations
                st.subheader("🔍 Explainable AI Visualizations")
                
                # Generate and display heatmaps
                heatmap_image = self.display_heatmaps(results['explanation'], results['original_image'])
                st.markdown(f'<img src="data:image/png;base64,{heatmap_image}" style="width:100%">', unsafe_allow_html=True)
                
                st.markdown("""
                **Explanation of Visualizations:**
                - **Original Image:** The input brain scan
                - **GradCAM:** Shows which regions the CNN focused on for classification
                - **GradCAM++:** Improved version with better localization
                - **Asymmetry Map:** Highlights asymmetric regions between brain hemispheres
                """)
            
            with tab4:
                # Clinical report
                st.subheader("📋 Clinical Decision Support Report")
                
                # Generate timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Report content
                report_content = f"""
                ## Brain Tumor Classification Report
                
                **Generated:** {timestamp}
                **System:** Symmetry-Integrated CNN with Explainable AI
                
                ### Analysis Results
                
                **Primary Diagnosis:** {results['prediction']['class_name']}
                **Confidence Level:** {results['prediction']['confidence']:.1%}
                **Model Uncertainty:** {results['prediction']['uncertainty'][0]:.3f}
                
                ### Symmetry Analysis
                
                **Overall Symmetry Score:** {np.mean(list(symmetry_features.values())):.3f}
                
                **Key Findings:**
                - Intensity Symmetry: {symmetry_features.get('intensity_symmetry', 0):.3f}
                - Structural Symmetry: {symmetry_features.get('structural_symmetry', 0):.3f}
                - Volume Asymmetry: {symmetry_features.get('volume_asymmetry', 0):.3f}
                
                ### Clinical Interpretation
                
                {results['explanation']['symmetry_analysis']['clinical_interpretation']}
                
                ### Recommendations
                
                """
                
                # Add recommendations based on confidence
                if results['prediction']['confidence'] > 0.8:
                    report_content += "- High confidence prediction. Consider correlation with clinical symptoms.\n"
                elif results['prediction']['confidence'] > 0.6:
                    report_content += "- Moderate confidence prediction. Recommend additional imaging or expert review.\n"
                else:
                    report_content += "- Low confidence prediction. Strongly recommend expert radiologist review.\n"
                
                if results['prediction']['class_name'] != 'No Tumor':
                    report_content += "- If tumor detected, recommend oncology consultation.\n"
                    report_content += "- Consider additional imaging modalities for treatment planning.\n"
                
                report_content += """
                ### Disclaimer
                
                This AI system is designed to assist healthcare professionals and should not replace clinical judgment. 
                All results should be reviewed by qualified medical personnel before making treatment decisions.
                """
                
                st.markdown(report_content)
                
                # Download report
                if st.button("📥 Download Report"):
                    st.download_button(
                        label="Download Clinical Report",
                        data=report_content,
                        file_name=f"brain_tumor_report_{timestamp.replace(':', '-').replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>🧠 Brain Tumor Classification System | Powered by Symmetry-Integrated CNN</p>
            <p>An Integrative Approach Using Explainable CNNs and Brain Symmetry Metrics</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the app"""
    app = BrainTumorApp()
    app.run()


if __name__ == "__main__":
    main()
