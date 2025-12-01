"""
Training Script for Brain Abnormality Autoencoder

Trains autoencoder ONLY on "No Tumor" images to learn normal brain patterns.
The trained model will then detect abnormalities by measuring reconstruction error.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.abnormality_autoencoder import BrainAbnormalityAutoencoder


class NormalBrainDataset(Dataset):
    """Dataset of ONLY normal brain images (No Tumor class)"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to data/raw/dataset directory
            transform: Image transformations
        """
        self.transform = transform
        
        # Collect all "No Tumor" images from Training and Testing
        self.image_paths = []
        
        # Training set
        train_no_tumor_dir = Path(data_dir) / 'Training' / 'no_tumor'
        if train_no_tumor_dir.exists():
            self.image_paths.extend(list(train_no_tumor_dir.glob('*.jpg')))
            self.image_paths.extend(list(train_no_tumor_dir.glob('*.png')))
        
        # Testing set
        test_no_tumor_dir = Path(data_dir) / 'Testing' / 'no_tumor'
        if test_no_tumor_dir.exists():
            self.image_paths.extend(list(test_no_tumor_dir.glob('*.jpg')))
            self.image_paths.extend(list(test_no_tumor_dir.glob('*.png')))
        
        print(f"Found {len(self.image_paths)} normal brain images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image


def train_autoencoder(
    data_dir='data/raw/dataset',
    output_dir='models',
    epochs=50,
    batch_size=16,
    learning_rate=0.001,
    device=None
):
    """
    Train the autoencoder on normal brain images
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on (cuda/cpu)
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = NormalBrainDataset(data_dir, transform=transform)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = BrainAbnormalityAutoencoder().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path(output_dir) / 'abnormality_autoencoder.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
            print(f'  ✅ Saved best model (val_loss: {val_loss:.6f})')
        
        print()
    
    print("=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {Path(output_dir) / 'abnormality_autoencoder.pth'}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path(output_dir) / 'autoencoder_training.png', dpi=150, bbox_inches='tight')
    print(f"Training plot saved to: {Path(output_dir) / 'autoencoder_training.png'}")
    
    return model, train_losses, val_losses


def test_autoencoder(model_path='models/abnormality_autoencoder.pth', data_dir='data/raw/dataset'):
    """
    Test the trained autoencoder on sample images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BrainAbnormalityAutoencoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Testing autoencoder on different tumor types...")
    print("=" * 60)
    
    # Test on one image from each class
    test_classes = [
        ('Testing/no_tumor', 'No Tumor'),
        ('Testing/glioma_tumor', 'Glioma Tumor'),
        ('Testing/meningioma_tumor', 'Meningioma Tumor'),
        ('Testing/pituitary_tumor', 'Pituitary Tumor'),
    ]
    
    for class_dir, class_name in test_classes:
        class_path = Path(data_dir) / class_dir
        if class_path.exists():
            # Get first image
            images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            if images:
                img_path = images[0]
                
                # Load and process image
                image = Image.open(img_path).convert('L')
                image = image.resize((224, 224))
                image_array = np.array(image)
                
                # Compute abnormality score
                score = model.compute_abnormality_score(image_array)
                
                print(f"{class_name:20s}: Abnormality Score = {score:.4f}")
    
    print("=" * 60)
    print("\nExpected behavior:")
    print("  - No Tumor:        LOW score (< 0.10)")
    print("  - Tumor classes:   HIGH score (> 0.15)")


if __name__ == "__main__":
    print("Brain Abnormality Autoencoder Training")
    print("=" * 60)
    print()
    
    # Train the model
    model, train_losses, val_losses = train_autoencoder(
        data_dir='data/raw/dataset',
        output_dir='models',
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print()
    
    # Test the model
    test_autoencoder()
    
    print("\n✅ All done! Autoencoder is ready to use.")
