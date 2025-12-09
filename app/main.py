"""
NeuroScan AI - Main Application Entry Point
Refactored, modular Streamlit application for brain tumor classification
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import UI_CONFIG, CLASS_DESCRIPTIONS
from app.services import (
    ModelService,
    SymmetryService,
    ExplanationService,
    ReportService
)
from app.components import ChartComponents, VisualizationComponents


def initialize_session_state():
    """Initialize session state variables"""
    if 'model_service' not in st.session_state:
        st.session_state.model_service = ModelService()
    if 'symmetry_service' not in st.session_state:
        st.session_state.symmetry_service = SymmetryService()
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None


def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title=UI_CONFIG['page_title'],
        page_icon=UI_CONFIG['page_icon'],
        layout=UI_CONFIG['layout'],
        initial_sidebar_state=UI_CONFIG['initial_sidebar_state']
    )


def render_header():
    """Render application header"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            🏥 {UI_CONFIG['app_name']}
        </h1>
        <p style="color: #f0f0f0; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            {UI_CONFIG['app_tagline']}
        </p>
        <p style="color: #d0d0d0; text-align: center; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
            v{UI_CONFIG['app_version']} | {UI_CONFIG['model_accuracy']} Accuracy | FDA Research Use Only
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with model loading and info"""
    with st.sidebar:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="color: white; text-align: center; margin: 0;">🏥 {UI_CONFIG['app_name']}</h3>
            <p style="color: #e0e0e0; text-align: center; margin: 0.3rem 0 0 0; font-size: 0.85rem;">
                v{UI_CONFIG['app_version']} Control Panel
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model loading
        st.subheader("🔧 Model Management")
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                success = st.session_state.model_service.load_model()
                if success:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model")
        
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Model not loaded")
        
        st.divider()
        
        # System info
        st.subheader("📋 System Information")
        device = st.session_state.model_service.device
        st.info(f"**Device:** {device}")
        st.info(f"**Classes:** 4 tumor types")
        
        # Class descriptions
        st.subheader("🏥 Tumor Types")
        for class_name, description in CLASS_DESCRIPTIONS.items():
            with st.expander(class_name):
                st.write(description)


def analyze_image(image: Image.Image):
    """
    Perform complete image analysis
    
    Args:
        image: PIL Image to analyze
    """
    model_service = st.session_state.model_service
    symmetry_service = st.session_state.symmetry_service
    
    # Preprocess image
    image_tensor, image_array = model_service.preprocess_image(image)
    
    # Get original image for symmetry
    original_image = np.array(image.resize((224, 224)))
    if len(original_image.shape) == 3:
        original_image = np.mean(original_image, axis=2)
    
    # Model prediction
    prediction_results = model_service.predict(image_tensor)
    
    # Symmetry analysis
    symmetry_results = symmetry_service.analyze(original_image)
    
    # Clinical interpretation
    tumor_detected = prediction_results['class_name'] != 'No Tumor'
    clinical_interpretation = symmetry_service.generate_clinical_interpretation(
        symmetry_results['features'],
        tumor_detected=tumor_detected,
        tumor_type=prediction_results['class_name']
    )
    
    # Initialize explanation service
    explanation_service = ExplanationService(model_service.get_model())
    
    # Generate visual explanations
    visual_explanations = explanation_service.create_visual_explanations(
        image_tensor,
        original_image,
        prediction_results['predicted_class'],
        symmetry_results['left_hemisphere'],
        symmetry_results['right_hemisphere']
    )
    
    # Store results
    st.session_state.analysis_results = {
        'prediction': prediction_results,
        'symmetry': symmetry_results,
        'clinical_interpretation': clinical_interpretation,
        'visual_explanations': visual_explanations,
        'original_image': original_image,
        'processed_image': image_array
    }


def render_analysis_page():
    """Render the main analysis page"""
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model using the sidebar controls before proceeding.")
        st.info("💡 The system will automatically look for the latest trained model in the results directory.")
        return
    
    # New analysis button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state.analysis_results = None
            st.rerun()
    
    st.markdown("---")
    st.header("📤 Upload Brain Scan")
    
    # Warning
    st.warning("⚠️ **IMPORTANT:** This system is trained ONLY on brain MRI/CT scans. Uploading other images will produce meaningless results!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a brain scan image...",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="⚠️ ONLY upload brain MRI or CT scan images"
    )
    
    if uploaded_file is not None:
        # Check if new file
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.analysis_results = None
            st.session_state.last_uploaded_file = uploaded_file.name
        
        # Load image with loading indicator
        with st.spinner("📤 Loading image..."):
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return
        
        # Two-column layout
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("### 📷 Uploaded Image")
            st.image(image, width=300)
            
            st.warning("⚠️ Ensure this is a brain MRI/CT scan!")
            
            if st.button("🔍 ANALYZE IMAGE", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing..."):
                    analyze_image(image)
                    if st.session_state.analysis_results:
                        pred = st.session_state.analysis_results['prediction']
                        if pred['confidence'] < 0.4:
                            st.error("❌ Very low confidence! This may not be a brain scan image.")
                        else:
                            st.success("✅ Analysis complete!")
                        st.rerun()
        
        with right_col:
            if st.session_state.analysis_results is not None:
                results = st.session_state.analysis_results
                prediction = results['prediction']
                
                # Determine styling
                confidence = prediction['confidence']
                class_name = prediction['class_name']
                
                if confidence > 0.8:
                    bg_color, border_color, icon, conf_label = "#d4edda", "#28a745", "🎯", "HIGH"
                elif confidence > 0.6:
                    bg_color, border_color, icon, conf_label = "#fff3cd", "#ffc107", "⚠️", "MEDIUM"
                else:
                    bg_color, border_color, icon, conf_label = "#f8d7da", "#dc3545", "❓", "LOW"
                
                st.markdown("### 🎯 Prediction Result")
                st.markdown(f"""
                <div style="background-color: {bg_color}; border-left: 8px solid {border_color}; 
                            padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                    <h2 style="margin: 0; color: #333; font-size: 2rem;">{icon} {class_name}</h2>
                    <h3 style="margin: 0.5rem 0 0 0; color: {border_color}; font-size: 1.5rem;">
                        {confidence:.1%}
                    </h3>
                    <p style="margin: 0.3rem 0 0 0; color: #666;">{conf_label} CONFIDENCE</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick metrics
                st.markdown("### 📊 Quick Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Class", class_name)
                    st.metric("Uncertainty", f"{prediction['uncertainty'][0].item():.3f}")
                with met_col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.metric("Model Score", f"{prediction['confidence_results']['confidence'][0].item():.1%}")
            else:
                st.info("👆 Click 'ANALYZE IMAGE' to get predictions")
    
    # Display detailed results
    if st.session_state.analysis_results is not None:
        st.divider()
        st.header("📈 Detailed Analysis")
        
        results = st.session_state.analysis_results
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Predictions", 
            "🧠 Symmetry Analysis", 
            "🔍 Visual Explanations", 
            "📊 Clinical Report"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction chart
                probabilities = results['prediction']['probabilities'][0].cpu().numpy()
                fig = ChartComponents.create_prediction_chart(probabilities)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence chart
                confidence_data = {
                    'mean': results['prediction']['confidence_results']['predictions'][0].cpu().numpy(),
                    'std': results['prediction']['confidence_results']['std'][0].cpu().numpy()
                }
                fig = ChartComponents.create_confidence_chart(confidence_data)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Radar chart (new!)
            st.subheader("🎯 Core 4 Symmetry Metrics Overview")
            fig_radar = ChartComponents.create_symmetry_radar_chart(results['symmetry']['features'])
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart
                fig = ChartComponents.create_symmetry_chart(results['symmetry']['features'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🧠 Metrics with Thresholds")
                for name, value in results['symmetry']['features'].items():
                    display_name = name.replace('_', ' ').title()
                    
                    # Determine status based on metric type
                    if 'abnormality' in name:
                        # Higher is worse for abnormality
                        if value < 0.05:
                            delta = "Normal"
                            delta_color = "normal"
                        elif value < 0.10:
                            delta = "Mild"
                            delta_color = "off"
                        else:
                            delta = "High"
                            delta_color = "inverse"
                    elif 'asymmetry' in name:
                        # Higher is worse for asymmetry
                        if value < 0.30:
                            delta = "Symmetric"
                            delta_color = "normal"
                        elif value < 0.50:
                            delta = "Moderate"
                            delta_color = "off"
                        else:
                            delta = "Asymmetric"
                            delta_color = "inverse"
                    else:
                        # Higher is better for balance metrics
                        if value > 0.70:
                            delta = "High Balance"
                            delta_color = "normal"
                        elif value > 0.50:
                            delta = "Moderate"
                            delta_color = "off"
                        else:
                            delta = "Low Balance"
                            delta_color = "inverse"
                    
                    st.metric(display_name, f"{value:.3f}", delta=delta, delta_color=delta_color)
            
            st.subheader("🏥 Clinical Interpretation")
            st.info(results['clinical_interpretation'])
            
            # Hemisphere comparison
            st.subheader("📊 Hemisphere Comparison")
            hemisphere_viz = VisualizationComponents.create_hemisphere_comparison(
                results['symmetry']['left_hemisphere'],
                results['symmetry']['right_hemisphere'],
                results['symmetry']['midline'],
                results['original_image']
            )
            st.markdown(
                f'<img src="data:image/png;base64,{hemisphere_viz}" style="width:100%">',
                unsafe_allow_html=True
            )
        
        with tab3:
            st.subheader("🔍 Explainable AI Visualizations")
            
            # Generate heatmaps
            heatmap_viz = VisualizationComponents.create_heatmap_visualization(
                results['original_image'],
                results['visual_explanations']
            )
            st.markdown(
                f'<img src="data:image/png;base64,{heatmap_viz}" style="width:100%">',
                unsafe_allow_html=True
            )
            
            # Explanation text
            st.markdown("### 📖 Understanding These Visualizations")
            st.info(f"""
            **Prediction: {results['prediction']['class_name']} ({results['prediction']['confidence']:.1%} confidence)**
            
            - **GradCAM Heatmaps (Red = High Attention):** Shows which regions the AI focused on
            - **Asymmetry Map (Bright = High Asymmetry):** Differences between left and right hemispheres
            """)
        
        with tab4:
            st.subheader("📋 Clinical Decision Support Report")
            
            # Generate report
            report = ReportService.generate_clinical_report(
                results['prediction'],
                results['symmetry']['features'],
                results['clinical_interpretation']
            )
            
            st.markdown(report)
            
            # Download button
            filename = ReportService.generate_filename()
            st.download_button(
                label="📥 Download Complete Report",
                data=report,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True
            )


def main():
    """Main application entry point"""
    setup_page()
    initialize_session_state()
    render_header()
    render_sidebar()
    render_analysis_page()


if __name__ == "__main__":
    main()
