"""
Pure CNN Training Script for Brain Tumor Classification
No symmetry integration - CNN visual features only
Simpler, more reliable, and clinically sound approach
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import BrainTumorDataLoader


class PureCNNModel(nn.Module):
    """
    Pure CNN model for brain tumor classification
    No symmetry features - just visual features from EfficientNet
    """
    
    def __init__(self, num_classes=4, backbone='efficientnet_b3', pretrained=True):
        super(PureCNNModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Create backbone
        if backbone == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            backbone_features = 1536
            # Remove the final classifier
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_features = 2048
            self.backbone.fc = nn.Identity()
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            backbone_features = 1024
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
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
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'architecture': 'Pure CNN (no symmetry)'
        }


class Trainer:
    """Training manager for pure CNN model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"training_run_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        
        # Initialize data loader
        self.data_loader = BrainTumorDataLoader(
            data_dir=config['data_dir'],
            image_size=tuple(config['image_size']),
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Setup data loaders
        self.data_loader.setup_data_loaders(
            test_size=config['test_size'],
            val_size=config['val_size'],
            random_state=config['random_state'],
            use_weighted_sampling=config['use_weighted_sampling']
        )
        
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.get_data_loaders()
        
        # Initialize model
        self.model = PureCNNModel(
            num_classes=4,
            backbone=config['backbone'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # Print model info
        model_info = self.model.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            if isinstance(value, int):
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value}")
        
        # Loss function
        if config['use_class_weights']:
            class_weights = self.data_loader.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        # Learning rate scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config['min_lr']
            )
        elif config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif config['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def test(self):
        """Test the model and generate detailed metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        
        # Classification report
        class_names = self.data_loader.class_names
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC AUC (one-vs-rest)
        all_probs = np.array(all_probs)
        all_labels_onehot = np.zeros((len(all_labels), len(class_names)))
        all_labels_onehot[np.arange(len(all_labels)), all_labels] = 1
        
        roc_auc = {}
        for i, class_name in enumerate(class_names):
            roc_auc[class_name] = roc_auc_score(
                all_labels_onehot[:, i],
                all_probs[:, i]
            )
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Test Accuracy: {accuracy:.2f}%")
        print("\nPer-Class Metrics:")
        for class_name in class_names:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  ROC-AUC: {roc_auc[class_name]:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names)
        
        # Save test results
        test_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_metrics.png', dpi=300)
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(
            checkpoint,
            self.output_dir / 'checkpoints' / 'latest.pth'
        )
        
        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint,
                self.output_dir / 'checkpoints' / 'best_model.pth'
            )
            print(f"✓ Saved best model (val_acc: {self.best_val_acc:.2f}%)")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.config['scheduler'] == 'plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Plot training history
        self.plot_training_history()
        
        # Test the best model
        print("\nLoading best model for testing...")
        checkpoint = torch.load(self.output_dir / 'checkpoints' / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = self.test()
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Test accuracy: {test_results['accuracy']:.2f}%")
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main training function"""
    
    # Configuration
    config = {
        # Data
        'data_dir': 'data/raw/dataset',
        'image_size': [224, 224],
        'batch_size': 32,
        'num_workers': 4,
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        'use_weighted_sampling': True,
        
        # Model
        'backbone': 'efficientnet_b3',
        'pretrained': True,
        'use_class_weights': True,
        
        # Training
        'epochs': 50,
        'learning_rate': 0.0001,
        'weight_decay': 0.00001,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'min_lr': 0.0000001,
        'grad_clip': 1.0,
        
        # Early stopping
        'patience': 10,
        'min_delta': 0.001,
        'save_every': 10,
        
        # Output
        'output_dir': 'results'
    }
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
