---
title: NeuroScan AI - Brain Tumor Classifier
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app/main.py
pinned: false
license: mit
---

# 🧠 NeuroScan AI - Clinical Decision Support System

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/YOUR_USERNAME/neuroscan-ai)

## 🎯 Overview

NeuroScan AI is an advanced **Clinical Decision Support System** for brain tumor detection and classification. This system combines state-of-the-art deep learning with brain symmetry analysis to provide accurate, explainable predictions for medical professionals.

### Key Features
- 🔬 **4-Class Classification**: Glioma, Meningioma, Pituitary, No Tumor
- 🧠 **Brain Symmetry Analysis**: Quantitative asymmetry detection
- 🔍 **Explainable AI**: GradCAM heatmaps and visual explanations
- 📊 **Clinical Reports**: Automated diagnostic report generation
- ⚡ **95.82% Accuracy**: Validated on comprehensive test dataset

## 🚀 How to Use

1. **Load the Model**: Click "🔄 Load Model" in the sidebar
2. **Upload a Brain Scan**: Use the file uploader (MRI/CT scans only)
3. **Analyze**: Click "🔍 ANALYZE IMAGE" to get predictions
4. **Review Results**: Explore predictions, symmetry analysis, and explanations
5. **Download Report**: Generate a clinical decision support report

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is designed for **research and educational purposes only**. It should NOT be used as the sole basis for medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 🏗️ Technical Details

### Model Architecture
- **Backbone**: EfficientNet-B3 (pre-trained on ImageNet)
- **Symmetry Analysis**: 8 quantitative brain symmetry metrics
- **Explainability**: GradCAM and GradCAM++ for visual explanations
- **Uncertainty**: Monte Carlo dropout for confidence estimation

### Performance Metrics
- **Accuracy**: 95.82% on test set
- **Precision**: 93.8% (macro average)
- **Recall**: 94.1% (macro average)
- **F1-Score**: 93.9% (macro average)

### Tumor Types Detected
1. **Glioma Tumor**: Tumors arising from glial cells in brain/spinal cord
2. **Meningioma Tumor**: Tumors of the meninges (usually benign)
3. **Pituitary Tumor**: Abnormal growth in pituitary gland tissues
4. **No Tumor**: Normal brain tissue with no detectable tumor

## 📊 Symmetry Analysis Features

The system analyzes brain symmetry across multiple dimensions:
- **Intensity Balance**: Pixel intensity comparison between hemispheres
- **Structural Balance**: Edge and contour symmetry
- **Asymmetry Index**: Overall quantification of brain asymmetry
- **Abnormality Score**: Detection of unusual patterns

## 🔍 Explainable AI Components

- **GradCAM Heatmaps**: Shows which brain regions the AI focused on
- **Asymmetry Maps**: Visualizes left-right hemisphere differences
- **Feature Attribution**: Quantitative importance of each input feature
- **Clinical Interpretation**: Automated text explanations

## 📈 Use Cases

- **Clinical Decision Support**: Assist radiologists in tumor detection
- **Educational Tool**: Training medical students on tumor characteristics
- **Research Platform**: Validate new imaging techniques
- **Triage System**: Prioritize urgent cases for expert review

## 🛠️ Technology Stack

- **Framework**: PyTorch, Streamlit
- **Visualization**: Plotly, Matplotlib
- **Explainability**: Captum (GradCAM)
- **Image Processing**: OpenCV, PIL, scikit-image

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{neuroscan_ai_2024,
  title={NeuroScan AI: Clinical Decision Support System for Brain Tumor Detection},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/spaces/YOUR_USERNAME/neuroscan-ai}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Brain Tumor Classification Dataset
- Pre-trained Models: EfficientNet (ImageNet)
- Frameworks: PyTorch, Streamlit, Hugging Face

## 📞 Contact

For questions, feedback, or collaboration opportunities, please reach out through the Community tab or visit the [GitHub repository](https://github.com/Edmond-Amofah-Boakye/brain_tumor-prediction).

---

**Built with ❤️ for advancing medical AI and improving patient outcomes**
