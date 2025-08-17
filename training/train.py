"""
Training Pipeline for Symmetry-Integrated CNN
Comprehensive training script with monitoring, checkpointing, and evaluation
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.symmetry_cnn import create_symmetry_cnn, SymmetryLoss
from models.symmetry_analyzer import BrainSymmetryAnalyzer
from data.data_loader import BrainTumorDataLoader
from explainability.gradcam import create_explainer


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Acc', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.learning_rates, color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Epoch time plot
        axes[1, 1].plot(self.epoch_times, color='orange')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig


class BrainTumorTrainer:
    """Main trainer class for brain tumor classification"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        self.output_dir = Path(self.config['output_dir'])
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.output_dir, self.checkpoints_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        self.data_loader = BrainTumorDataLoader(
            data_dir=self.config['data_dir'],
            image_size=tuple(self.config['image_size']),
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4)
        )
        
        self.data_loader.setup_data_loaders(
            test_size=self.config.get('test_size', 0.2),
            val_size=self.config.get('val_size', 0.2),
            random_state=self.config.get('random_state', 42),
            use_weighted_sampling=self.config.get('use_weighted_sampling', True)
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_data_loaders()
        self.class_info = self.data_loader.get_class_info()
        
        print(f"Data loaded successfully!")
        print(f"Train samples: {len(self.data_loader.train_dataset)}")
        print(f"Val samples: {len(self.data_loader.val_dataset)}")
        print(f"Test samples: {len(self.data_loader.test_dataset)}")
    
    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function"""
        print("Setting up model...")
        
        # Create model
        self.model = create_symmetry_cnn(
            num_classes=self.class_info['num_classes'],
            backbone=self.config.get('backbone', 'efficientnet_b3'),
            pretrained=self.config.get('pretrained', True),
            freeze_backbone=self.config.get('freeze_backbone', False)
        ).to(self.device)
        
        # Print model info
        model_info = self.model.get_model_info()
        print(f"Model created with {model_info['total_parameters']:,} parameters")
        print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Setup optimizer
        optimizer_name = self.config.get('optimizer', 'adamw')
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup scheduler
        scheduler_name = self.config.get('scheduler', 'cosine')
        if scheduler_name.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('min_lr', 1e-7)
            )
        elif scheduler_name.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        # Setup loss function
        if self.config.get('use_class_weights', True):
            class_weights = self.class_info['class_weights'].to(self.device)
        else:
            class_weights = None
        
        self.criterion = SymmetryLoss(
            alpha=self.config.get('classification_weight', 1.0),
            beta=self.config.get('symmetry_weight', 0.1)
        )
        
        # If using class weights, modify the criterion
        if class_weights is not None:
            self.criterion.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, uncertainty = self.model(images)
            
            # Calculate loss
            loss_dict = self.criterion(logits, labels, None, uncertainty)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                logits, uncertainty = self.model(images)
                
                # Calculate loss
                loss_dict = self.criterion(logits, labels, None, uncertainty)
                loss = loss_dict['total_loss']
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {self.config['epochs']} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc, val_predictions, val_labels = self.validate_epoch()
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics_tracker.update(
                train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time
            )
            
            # Print epoch results
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'LR: {current_lr:.2e}, Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f'New best validation accuracy: {val_acc:.2f}%')
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/3600:.2f} hours')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        
        # Plot training metrics
        self.metrics_tracker.plot_metrics(self.plots_dir / 'training_metrics.png')
        
        # Save training history
        self.save_training_history()
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("Evaluating on test set...")
        
        # Load best model
        best_model_path = self.checkpoints_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                logits, uncertainty = self.model(images)
                probabilities = torch.softmax(logits, dim=1)
                
                # Predictions
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        test_acc = 100. * correct / total
        print(f'Test Accuracy: {test_acc:.2f}%')
        
        # Generate detailed evaluation report
        self.generate_evaluation_report(all_labels, all_predictions, all_probabilities)
        
        return test_acc
    
    def generate_evaluation_report(self, true_labels, predictions, probabilities):
        """Generate comprehensive evaluation report"""
        class_names = self.class_info['class_names']
        
        # Classification report
        report = classification_report(
            true_labels, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Save classification report
        with open(self.logs_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC AUC (for multiclass)
        try:
            probabilities_array = np.array(probabilities)
            auc_scores = {}
            for i, class_name in enumerate(class_names):
                # One-vs-rest AUC
                binary_labels = (np.array(true_labels) == i).astype(int)
                class_probs = probabilities_array[:, i]
                auc = roc_auc_score(binary_labels, class_probs)
                auc_scores[class_name] = auc
            
            print("\nAUC Scores (One-vs-Rest):")
            for class_name, auc in auc_scores.items():
                print(f"{class_name}: {auc:.4f}")
            
            # Save AUC scores
            with open(self.logs_dir / 'auc_scores.json', 'w') as f:
                json.dump(auc_scores, f, indent=2)
                
        except Exception as e:
            print(f"Could not calculate AUC scores: {e}")
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'config': self.config
        }
        
        torch.save(checkpoint, self.checkpoints_dir / filename)
    
    def save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': self.metrics_tracker.train_losses,
            'val_losses': self.metrics_tracker.val_losses,
            'train_accuracies': self.metrics_tracker.train_accuracies,
            'val_accuracies': self.metrics_tracker.val_accuracies,
            'learning_rates': self.metrics_tracker.learning_rates,
            'epoch_times': self.metrics_tracker.epoch_times,
            'best_val_acc': self.best_val_acc
        }
        
        with open(self.logs_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_config():
    """Create default training configuration"""
    config = {
        # Data settings
        'data_dir': 'data/raw/dataset',
        'image_size': [224, 224],
        'batch_size': 32,
        'num_workers': 4,
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        'use_weighted_sampling': True,
        
        # Model settings
        'backbone': 'efficientnet_b3',
        'pretrained': True,
        'freeze_backbone': False,
        'use_class_weights': True,
        
        # Training settings
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'min_lr': 1e-7,
        'grad_clip': 1.0,
        
        # Loss settings
        'classification_weight': 1.0,
        'symmetry_weight': 0.1,
        
        # Early stopping
        'patience': 10,
        'min_delta': 0.001,
        
        # Checkpointing
        'save_every': 10,
        'output_dir': 'results/training_run_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/raw/dataset', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3', help='Model backbone')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_config()
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    if args.backbone:
        config['backbone'] = args.backbone
    
    # Save configuration
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Training Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Initialize trainer
    trainer = BrainTumorTrainer(config)
    
    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Train model
    trainer.train()
    
    # Evaluate model
    trainer.evaluate()
    
    print(f"Training completed! Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
