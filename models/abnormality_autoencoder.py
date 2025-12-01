"""
Autoencoder for Brain Tissue Abnormality Detection

This autoencoder is trained ONLY on normal brain images ("No Tumor" class).
It learns what normal brain tissue looks like, then detects abnormalities
by measuring reconstruction error on new images.

High reconstruction error = Image doesn't match learned "normal" pattern = Abnormal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BrainAbnormalityAutoencoder(nn.Module):
    """
    Lightweight autoencoder for detecting abnormal brain tissue.
    
    Architecture:
    - Encoder: Compress 224x224 image to 7x7x32 latent space
    - Decoder: Reconstruct 224x224 image from latent space
    
    Training: Only on "No Tumor" images
    Inference: Reconstruction error indicates abnormality
    """
    
    def __init__(self):
        super(BrainAbnormalityAutoencoder, self).__init__()
        
        # Encoder: 224x224 -> 7x7x32
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 56 -> 28
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            # 28 -> 14
            nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14 -> 7 (latent space)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: 7x7x32 -> 224x224
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 14 -> 28
            nn.ConvTranspose2d(32, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            # 28 -> 56
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 56 -> 112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 112 -> 224
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output [0,1] range
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image [B, 1, 224, 224]
            
        Returns:
            reconstructed: Reconstructed image [B, 1, 224, 224]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def compute_abnormality_score(self, image):
        """
        Compute abnormality score for a single image
        
        Args:
            image: numpy array [224, 224] or [1, 224, 224]
            
        Returns:
            abnormality_score: Float 0-1 (higher = more abnormal)
        """
        import numpy as np
        
        # Prepare image
        if len(image.shape) == 2:
            image = image[np.newaxis, :]  # Add channel dim
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        # Normalize to [0,1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # [1, 1, 224, 224]
        
        # Get reconstruction
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(image)
        
        # Compute reconstruction error (MSE)
        mse = F.mse_loss(reconstructed, image, reduction='mean').item()
        
        # Scale MSE to abnormality score (0-1 range)
        # Typical MSE for normal brains: 0.005-0.015
        # Typical MSE for tumor brains: 0.020-0.050+
        # Scale accordingly
        abnormality_score = min(1.0, mse * 20.0)  # Scale factor tuned experimentally
        
        return abnormality_score
    
    def get_reconstruction_error_map(self, image):
        """
        Get spatial map of reconstruction errors
        
        Args:
            image: numpy array [224, 224]
            
        Returns:
            error_map: numpy array [224, 224] showing per-pixel errors
        """
        import numpy as np
        
        # Prepare image
        if len(image.shape) == 2:
            image = image[np.newaxis, :]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Get reconstruction
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(image)
        
        # Compute per-pixel error
        error_map = torch.abs(reconstructed - image).squeeze().cpu().numpy()
        
        return error_map


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("Testing Brain Abnormality Autoencoder...")
    
    # Create model
    model = BrainAbnormalityAutoencoder()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    test_input = torch.randn(2, 1, 224, 224)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test abnormality score computation
    import numpy as np
    test_image = np.random.rand(224, 224).astype(np.float32)
    score = model.compute_abnormality_score(test_image)
    print(f"Abnormality score: {score:.4f}")
    
    print("\n✅ Model test passed!")
