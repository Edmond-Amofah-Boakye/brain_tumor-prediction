# Brain Tumor Diagnosis Using Explainable CNNs and Brain Symmetry Metrics

## 🎯 Project Overview

**Title**: "An Integrative Approach to Brain Tumor Diagnosis Using Explainable Convolutional Neural Networks and Brain Symmetry Metrics"

This project combines three powerful approaches for brain tumor diagnosis:
1. **Deep Learning**: Convolutional Neural Networks for tumor classification
2. **Explainable AI**: Grad-CAM visualizations for model interpretability
3. **Brain Symmetry Analysis**: Asymmetry detection as additional diagnostic feature

## 🧠 Problem Statement

Brain tumor diagnosis from MRI scans requires:
- High accuracy classification (No Tumor, Glioma, Meningioma, Pituitary)
- Explainable predictions for clinical trust
- Additional validation through anatomical analysis

## 🔬 Methodology

### 1. CNN Classification
- Multi-class classification (4 classes)
- Transfer learning or custom architecture
- Data augmentation and regularization

### 2. Explainable AI
- Grad-CAM heatmap generation
- Visual explanation of model decisions
- Overlay visualizations on original MRI

### 3. Brain Symmetry Analysis
- Left-right hemisphere comparison
- Asymmetry score calculation
- Structural similarity metrics

## 📁 Project Structure

```
brain_tumor_analysis/
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Preprocessed images
│   └── splits/              # Train/val/test splits
├── models/
│   ├── cnn_model.py         # CNN architecture
│   ├── symmetry_analyzer.py # Brain symmetry analysis
│   └── explainer.py         # Grad-CAM implementation
├── utils/
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py     # Image preprocessing
│   └── visualization.py     # Plotting functions
├── results/
│   ├── models/              # Saved model weights
│   ├── plots/               # Generated visualizations
│   └── reports/             # Analysis reports
├── main.py                  # Main execution script
└── config.py                # Configuration settings
```

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download brain tumor dataset from Kaggle
2. Place in `data/raw/` directory
3. Run preprocessing pipeline

### Training
```bash
python main.py --mode train
```

### Evaluation
```bash
python main.py --mode evaluate
```

## 📊 Expected Results

- **Classification Accuracy**: >90% on test set
- **Explainability**: Clear Grad-CAM visualizations
- **Symmetry Analysis**: Correlation with tumor presence
- **Clinical Relevance**: High sensitivity/specificity

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Computer Vision**: OpenCV, scikit-image
- **Explainable AI**: tf-explain, Grad-CAM
- **Visualization**: Matplotlib, Plotly
- **Data Processing**: Pandas, NumPy

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix Analysis
- ROC-AUC Curves
- Symmetry Score Correlation
- Grad-CAM Relevance Assessment

## 🎓 Academic Contribution

This project demonstrates:
- Integration of multiple AI approaches
- Medical domain knowledge incorporation
- Explainable AI for healthcare applications
- Novel symmetry-based validation method

---
*Master's Thesis Project - Brain Tumor Diagnosis*
