"""
Demo Script for Brain Tumor Classification System
Simple demonstration of the core functionality
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.symmetry_analyzer import BrainSymmetryAnalyzer
from models.symmetry_cnn import create_symmetry_cnn
from data.data_loader import BrainTumorDataLoader


def create_demo_image():
    """Create a synthetic brain scan image for demonstration"""
    # Create a synthetic brain-like image
    image = np.zeros((224, 224), dtype=np.uint8)
    
    # Create brain outline (ellipse)
    center_x, center_y = 112, 112
    for y in range(224):
        for x in range(224):
            # Ellipse equation
            if ((x - center_x) / 80) ** 2 + ((y - center_y) / 100) ** 2 <= 1:
                # Add some texture
                image[y, x] = 100 + np.random.randint(-30, 30)
    
    # Add some asymmetry to simulate a tumor
    # Create a bright spot on one side
    tumor_x, tumor_y = 140, 100
    for y in range(max(0, tumor_y-15), min(224, tumor_y+15)):
        for x in range(max(0, tumor_x-15), min(224, tumor_x+15)):
            if ((x - tumor_x) / 15) ** 2 + ((y - tumor_y) / 15) ** 2 <= 1:
                image[y, x] = min(255, image[y, x] + 80)
    
    # Convert to PIL Image
    return Image.fromarray(image).convert('RGB')


def demo_symmetry_analysis():
    """Demonstrate brain symmetry analysis"""
    print("="*60)
    print("BRAIN SYMMETRY ANALYSIS DEMO")
    print("="*60)
    
    # Create demo image
    demo_image = create_demo_image()
    
    # Initialize symmetry analyzer
    analyzer = BrainSymmetryAnalyzer(image_size=(224, 224))
    
    # Extract symmetry features
    print("Extracting symmetry features...")
    features = analyzer.extract_all_symmetry_features(np.array(demo_image))
    
    # Display results
    print("\nSymmetry Analysis Results:")
    print("-" * 40)
    for feature_name, value in features.items():
        print(f"{feature_name:25}: {value:.4f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = analyzer.visualize_symmetry_analysis()
    plt.savefig('demo_symmetry_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'demo_symmetry_analysis.png'")
    
    return features


def demo_model_architecture():
    """Demonstrate model architecture"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE DEMO")
    print("="*60)
    
    # Create model
    print("Creating Symmetry-Integrated CNN...")
    model = create_symmetry_cnn(
        num_classes=4,
        backbone='efficientnet_b3',
        pretrained=True
    )
    
    # Get model information
    model_info = model.get_model_info()
    print("\nModel Information:")
    print("-" * 40)
    for key, value in model_info.items():
        if isinstance(value, int):
            print(f"{key:25}: {value:,}")
        else:
            print(f"{key:25}: {value}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        logits, uncertainty = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Uncertainty shape: {uncertainty.shape}")
    
    # Test prediction with confidence
    print("\nTesting prediction with confidence...")
    results = model.predict_with_confidence(dummy_input[:1], num_samples=5)
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Confidence shape: {results['confidence'].shape}")
    
    return model


def demo_data_loading():
    """Demonstrate data loading (if dataset exists)"""
    print("\n" + "="*60)
    print("DATA LOADING DEMO")
    print("="*60)
    
    data_dir = "data/raw/dataset"
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found.")
        print("Creating synthetic dataset structure for demonstration...")
        
        # Create directory structure
        os.makedirs(f"{data_dir}/Training/glioma_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/Training/meningioma_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/Training/no_tumor", exist_ok=True)
        os.makedirs(f"{data_dir}/Training/pituitary_tumor", exist_ok=True)
        
        # Create a few demo images
        demo_image = create_demo_image()
        for class_name in ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']:
            for i in range(3):
                demo_image.save(f"{data_dir}/Training/{class_name}/demo_{i}.jpg")
        
        print("Synthetic dataset created!")
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = BrainTumorDataLoader(
        data_dir=data_dir,
        image_size=(224, 224),
        batch_size=8
    )
    
    try:
        # Setup data loaders
        data_loader.setup_data_loaders(test_size=0.3, val_size=0.3)
        
        # Get class information
        class_info = data_loader.get_class_info()
        print("\nClass Information:")
        print("-" * 40)
        for key, value in class_info.items():
            if key != 'class_weights':
                print(f"{key:15}: {value}")
            else:
                print(f"{key:15}: {value.numpy()}")
        
        # Test data loading
        train_loader, val_loader, test_loader = data_loader.get_data_loaders()
        
        print(f"\nDataset sizes:")
        print(f"Training: {len(data_loader.train_dataset)}")
        print(f"Validation: {len(data_loader.val_dataset)}")
        print(f"Testing: {len(data_loader.test_dataset)}")
        
        # Test batch loading
        print("\nTesting batch loading...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images {images.shape}, Labels {labels.shape}")
            if batch_idx >= 1:  # Just test 2 batches
                break
        
        return data_loader
        
    except Exception as e:
        print(f"Error in data loading: {e}")
        return None


def demo_integration():
    """Demonstrate full system integration"""
    print("\n" + "="*60)
    print("FULL SYSTEM INTEGRATION DEMO")
    print("="*60)
    
    # Create demo image
    demo_image = create_demo_image()
    
    # Save demo image
    demo_image.save('demo_brain_scan.jpg')
    print("Demo brain scan saved as 'demo_brain_scan.jpg'")
    
    # Initialize components
    print("Initializing system components...")
    
    # Symmetry analyzer
    symmetry_analyzer = BrainSymmetryAnalyzer(image_size=(224, 224))
    
    # Model (without training)
    model = create_symmetry_cnn(num_classes=4, pretrained=True)
    model.eval()
    
    # Preprocess image
    print("Preprocessing image...")
    image_array = np.array(demo_image.resize((224, 224)))
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float() / 255.0
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0)
    
    # Model prediction
    print("Making prediction...")
    with torch.no_grad():
        logits, uncertainty = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Symmetry analysis
    print("Analyzing brain symmetry...")
    symmetry_features = symmetry_analyzer.extract_all_symmetry_features(
        np.array(demo_image.convert('L'))
    )
    
    # Display results
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    print("\nPrediction Results:")
    print("-" * 40)
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Uncertainty: {uncertainty[0].item():.4f}")
    
    print("\nClass Probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
        print(f"{class_name:12}: {prob.item():.1%}")
    
    print("\nSymmetry Analysis:")
    print("-" * 40)
    overall_symmetry = np.mean(list(symmetry_features.values()))
    print(f"Overall Symmetry: {overall_symmetry:.4f}")
    
    # Top 3 most asymmetric features
    sorted_features = sorted(symmetry_features.items(), key=lambda x: x[1])
    print("\nMost Asymmetric Features:")
    for feature_name, value in sorted_features[:3]:
        print(f"{feature_name:20}: {value:.4f}")
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'symmetry_features': symmetry_features,
        'class_names': class_names
    }


def main():
    """Main demo function"""
    print("🧠 BRAIN TUMOR CLASSIFICATION SYSTEM DEMO")
    print("An Integrative Approach Using Explainable CNNs and Brain Symmetry Metrics")
    print("=" * 80)
    
    try:
        # Demo 1: Symmetry Analysis
        symmetry_features = demo_symmetry_analysis()
        
        # Demo 2: Model Architecture
        model = demo_model_architecture()
        
        # Demo 3: Data Loading
        data_loader = demo_data_loading()
        
        # Demo 4: Full Integration
        results = demo_integration()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        
        print("\nGenerated Files:")
        print("- demo_symmetry_analysis.png (Symmetry analysis visualization)")
        print("- demo_brain_scan.jpg (Synthetic brain scan)")
        
        print("\nNext Steps:")
        print("1. Add your real brain tumor dataset to 'data/raw/dataset/'")
        print("2. Train the model: python training/train.py")
        print("3. Run the web app: streamlit run app.py")
        
        print("\nFor more information, see README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your environment and dependencies.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
