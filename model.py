"""3D Vision Transformer for approximating fractal IFS transformations."""
import math
from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):
    """Extract 3D patches from a grid and embed them."""
    
    def __init__(
        self,
        grid_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
    ):
        """Initialize patch embedding layer.
        
        Args:
            grid_size: Size of input grid (assumed cubic)
            patch_size: Size of each patch (assumed cubic)
            embed_dim: Dimension of patch embeddings
        """
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        assert grid_size % patch_size == 0, "Grid size must be divisible by patch size"
        
        self.num_patches_per_dim = grid_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 3
        self.patch_volume = patch_size ** 3
        
        # Linear projection of flattened patches
        self.proj = nn.Linear(self.patch_volume, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and embed patches.
        
        Args:
            x: Input grid of shape (B, D, H, W) where D=H=W=grid_size
            
        Returns:
            Embedded patches of shape (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        
        # Reshape to extract patches
        # (B, D, H, W) -> (B, n, p, n, p, n, p) where n=num_patches_per_dim, p=patch_size
        x = x.reshape(
            B,
            self.num_patches_per_dim, self.patch_size,
            self.num_patches_per_dim, self.patch_size,
            self.num_patches_per_dim, self.patch_size,
        )
        
        # Reorder to group patches
        # (B, n, p, n, p, n, p) -> (B, n, n, n, p, p, p)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        
        # Flatten patches
        # (B, n, n, n, p, p, p) -> (B, n*n*n, p*p*p)
        x = x.reshape(B, self.num_patches, self.patch_volume)
        
        # Project to embedding dimension
        x = self.proj(x)  # (B, num_patches, embed_dim)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention."""
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class GridReconstructor(nn.Module):
    """Reconstruct 3D grid from patch embeddings."""
    
    def __init__(
        self,
        embed_dim: int = 128,
        grid_size: int = 32,
        patch_size: int = 4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches_per_dim = grid_size // patch_size
        self.patch_volume = patch_size ** 3
        
        # Project embeddings back to patch values
        self.proj = nn.Linear(embed_dim, self.patch_volume)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct grid from embeddings.
        
        Args:
            x: Patch embeddings of shape (B, num_patches, embed_dim)
            
        Returns:
            Reconstructed grid of shape (B, D, H, W)
        """
        B = x.shape[0]
        
        # Project to patch values
        x = self.proj(x)  # (B, num_patches, patch_volume)
        
        # Reshape to patch layout
        n = self.num_patches_per_dim
        p = self.patch_size
        x = x.reshape(B, n, n, n, p, p, p)
        
        # Reorder to grid layout
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        
        # Flatten to final grid
        x = x.reshape(B, self.grid_size, self.grid_size, self.grid_size)
        
        return x


class FractalTransformer3D(nn.Module):
    """3D Vision Transformer for learning fractal IFS transformations."""
    
    def __init__(
        self,
        grid_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """Initialize fractal transformer.
        
        Args:
            grid_size: Size of input grid (assumed cubic)
            patch_size: Size of each patch (assumed cubic)
            embed_dim: Dimension of patch embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed dim
            dropout: Dropout rate
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(grid_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Grid reconstruction
        self.reconstructor = GridReconstructor(embed_dim, grid_size, patch_size)
        
        # Initialize positional embeddings
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        # Initialize positional embeddings with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other parameters
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            x: Input grid of shape (B, D, H, W)
            
        Returns:
            Transformed grid of shape (B, D, H, W)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output normalization
        x = self.norm(x)
        
        # Reconstruct grid
        x = self.reconstructor(x)
        
        return x
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    grid_size: int = 32,
    patch_size: int = 4,
    embed_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
) -> FractalTransformer3D:
    """Create a fractal transformer model with default settings.
    
    Args:
        grid_size: Size of input grid
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        
    Returns:
        Initialized model
    """
    model = FractalTransformer3D(
        grid_size=grid_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    print(f"Created FractalTransformer3D:")
    print(f"  Grid size: {grid_size}³")
    print(f"  Patch size: {patch_size}³")
    print(f"  Number of patches: {model.patch_embed.num_patches}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Transformer layers: {num_layers}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing FractalTransformer3D...")
    
    model = create_model()
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 32, 32, 32)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print(f"\nTest passed!")

