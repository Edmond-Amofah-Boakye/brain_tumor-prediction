"""
Data Loading and Preprocessing Pipeline for Brain Tumor Classification
Handles dataset loading, preprocessing, and augmentation
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BrainTumorDataset(Dataset):
    """
    Custom Dataset class for brain tumor images
    """
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 transform=None, class_to_idx: Optional[Dict] = None):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Image transformations
            class_to_idx: Mapping from class names to indices
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Create class mapping if not provided
        if class_to_idx is None:
            unique_labels = sorted(list(set(labels)))
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        
        # Convert labels to indices
        self.label_indices = [self.class_to_idx[label] for label in labels]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Get label
        label = self.label_indices[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return Image.new('RGB', (224, 224), (0, 0, 0))
    
    def get_class_distribution(self):
        """Get class distribution statistics"""
        unique, counts = np.unique(self.label_indices, return_counts=True)
        distribution = {}
        for idx, count in zip(unique, counts):
            class_name = self.idx_to_class[idx]
            distribution[class_name] = count
        return distribution
    
    def visualize_samples(self, num_samples=8, save_path=None):
        """
        Visualize random samples from dataset
        
        Args:
            num_samples: Number of samples to visualize
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Get random indices
        indices = np.random.choice(len(self), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image, label = self[idx]
            
            # Convert tensor to numpy if needed
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # CHW format
                    image = image.permute(1, 2, 0)
                image = image.numpy()
            
            # Normalize for display
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            axes[i].imshow(image)
            axes[i].set_title(f'Class: {self.idx_to_class[label]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class BrainTumorDataLoader:
    """
    Data loader manager for brain tumor dataset
    """
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32, num_workers: int = 4):
        """
        Initialize data loader
        
        Args:
            data_dir: Root directory containing the dataset
            image_size: Target image size (height, width)
            batch_size: Batch size for training
            num_workers: Number of worker processes
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Dataset information
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        # Data storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Class weights for handling imbalance
        self.class_weights = None
        
    def load_data_paths(self) -> Tuple[List[str], List[str]]:
        """
        Load all image paths and labels from directory structure
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # Check for Training and Testing directories
        train_dir = os.path.join(self.data_dir, 'Training')
        test_dir = os.path.join(self.data_dir, 'Testing')
        
        for split_dir in [train_dir, test_dir]:
            if os.path.exists(split_dir):
                for class_name in self.class_names:
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.exists(class_dir):
                        for filename in os.listdir(class_dir):
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                image_path = os.path.join(class_dir, filename)
                                image_paths.append(image_path)
                                labels.append(class_name)
        
        print(f"Loaded {len(image_paths)} images from {len(set(labels))} classes")
        return image_paths, labels
    
    def create_transforms(self):
        """Create data augmentation transforms"""
        
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((self.image_size[0] + 32, self.image_size[1] + 32)),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/Test transforms without augmentation
        val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def compute_class_weights(self, labels: List[str]) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced dataset
        
        Args:
            labels: List of class labels
            
        Returns:
            Class weights tensor
        """
        # Convert labels to indices
        label_indices = [self.class_to_idx[label] for label in labels]
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(label_indices),
            y=label_indices
        )
        
        return torch.FloatTensor(class_weights)
    
    def create_weighted_sampler(self, labels: List[str]) -> WeightedRandomSampler:
        """
        Create weighted sampler for balanced training
        
        Args:
            labels: List of class labels
            
        Returns:
            WeightedRandomSampler
        """
        # Convert labels to indices
        label_indices = [self.class_to_idx[label] for label in labels]
        
        # Compute sample weights
        class_counts = np.bincount(label_indices)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in label_indices]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def setup_data_loaders(self, test_size=0.2, val_size=0.2, random_state=42, 
                          use_weighted_sampling=True):
        """
        Setup train, validation, and test data loaders
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed
            use_weighted_sampling: Whether to use weighted sampling for training
        """
        # Load data paths
        image_paths, labels = self.load_data_paths()
        
        # Create transforms
        train_transforms, val_transforms = self.create_transforms()
        
        # Split data into train and test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=test_size, 
            stratify=labels, random_state=random_state
        )
        
        # Split training data into train and validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size,
            stratify=train_labels, random_state=random_state
        )
        
        # Create datasets
        self.train_dataset = BrainTumorDataset(
            train_paths, train_labels, train_transforms, self.class_to_idx
        )
        
        self.val_dataset = BrainTumorDataset(
            val_paths, val_labels, val_transforms, self.class_to_idx
        )
        
        self.test_dataset = BrainTumorDataset(
            test_paths, test_labels, val_transforms, self.class_to_idx
        )
        
        # Compute class weights
        self.class_weights = self.compute_class_weights(train_labels)
        
        # Create samplers
        train_sampler = None
        if use_weighted_sampling:
            train_sampler = self.create_weighted_sampler(train_labels)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Print dataset statistics
        self.print_dataset_stats()
    
    def print_dataset_stats(self):
        """Print dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Image size: {self.image_size}")
        print(f"Batch size: {self.batch_size}")
        
        print(f"\nDataset sizes:")
        print(f"Training: {len(self.train_dataset)}")
        print(f"Validation: {len(self.val_dataset)}")
        print(f"Testing: {len(self.test_dataset)}")
        
        print(f"\nClass distribution (Training):")
        train_dist = self.train_dataset.get_class_distribution()
        for class_name, count in train_dist.items():
            percentage = (count / len(self.train_dataset)) * 100
            print(f"{class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nClass weights: {self.class_weights.numpy()}")
        print("="*50)
    
    def visualize_class_distribution(self, save_path=None):
        """
        Visualize class distribution across splits
        
        Args:
            save_path: Path to save the plot
        """
        # Get distributions
        train_dist = self.train_dataset.get_class_distribution()
        val_dist = self.val_dataset.get_class_distribution()
        test_dist = self.test_dataset.get_class_distribution()
        
        # Create DataFrame for plotting
        data = []
        for class_name in self.class_names:
            data.append({
                'Class': class_name,
                'Split': 'Train',
                'Count': train_dist.get(class_name, 0)
            })
            data.append({
                'Class': class_name,
                'Split': 'Validation',
                'Count': val_dist.get(class_name, 0)
            })
            data.append({
                'Class': class_name,
                'Split': 'Test',
                'Count': test_dist.get(class_name, 0)
            })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Class', y='Count', hue='Split')
        plt.title('Class Distribution Across Data Splits')
        plt.xlabel('Tumor Class')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.legend(title='Data Split')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_sample_images(self, num_samples_per_class=2, save_path=None):
        """
        Visualize sample images from each class
        
        Args:
            num_samples_per_class: Number of samples to show per class
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(self.num_classes, num_samples_per_class, 
                                figsize=(num_samples_per_class * 4, self.num_classes * 3))
        
        if self.num_classes == 1:
            axes = axes.reshape(1, -1)
        if num_samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, class_name in enumerate(self.class_names):
            # Find samples of this class
            class_indices = [i for i, label in enumerate(self.train_dataset.label_indices) 
                           if label == class_idx]
            
            # Select random samples
            selected_indices = np.random.choice(class_indices, 
                                              min(num_samples_per_class, len(class_indices)), 
                                              replace=False)
            
            for sample_idx, data_idx in enumerate(selected_indices):
                image, label = self.train_dataset[data_idx]
                
                # Convert tensor to numpy for display
                if isinstance(image, torch.Tensor):
                    # Denormalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
                    
                    # Convert to numpy
                    image = image.permute(1, 2, 0).numpy()
                
                axes[class_idx, sample_idx].imshow(image)
                axes[class_idx, sample_idx].set_title(f'{class_name}')
                axes[class_idx, sample_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_data_loaders(self):
        """
        Get data loaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None:
            raise ValueError("Data loaders not set up. Call setup_data_loaders() first.")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_class_info(self):
        """
        Get class information
        
        Returns:
            Dictionary with class information
        """
        return {
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'num_classes': self.num_classes,
            'class_weights': self.class_weights
        }


# Utility functions
def create_cross_validation_splits(image_paths, labels, n_splits=5, random_state=42):
    """
    Create cross-validation splits for the dataset
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Convert labels to indices for stratification
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(labels)
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in skf.split(image_paths, label_indices):
        splits.append((train_idx, val_idx))
    
    return splits


def analyze_image_properties(data_dir):
    """
    Analyze properties of images in the dataset
    
    Args:
        data_dir: Directory containing the dataset
    """
    image_sizes = []
    image_channels = []
    image_formats = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path)
                    
                    image_sizes.append(image.size)  # (width, height)
                    image_channels.append(len(image.getbands()))
                    image_formats.append(image.format)
                    
                except Exception as e:
                    print(f"Error analyzing {image_path}: {e}")
    
    # Analyze results
    print(f"\nImage Analysis Results:")
    print(f"Total images analyzed: {len(image_sizes)}")
    
    if image_sizes:
        widths, heights = zip(*image_sizes)
        print(f"Image dimensions:")
        print(f"  Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}")
        print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}")
        
        print(f"Channels: {set(image_channels)}")
        print(f"Formats: {set(image_formats)}")


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = BrainTumorDataLoader(
        data_dir="data/raw/dataset",
        image_size=(224, 224),
        batch_size=32
    )
    
    # Setup data loaders
    data_loader.setup_data_loaders(test_size=0.2, val_size=0.2)
    
    # Visualize data
    data_loader.visualize_class_distribution()
    data_loader.visualize_sample_images()
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Test data loading
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        if batch_idx >= 2:  # Just test a few batches
            break
