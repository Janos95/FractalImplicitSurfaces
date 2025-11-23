"""Train the 3D Vision Transformer to approximate fractal IFS transformations."""
import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import create_model, FractalTransformer3D


class FractalDataset(Dataset):
    """Dataset of random grids and their IFS transformations."""
    
    def __init__(self, data_path: Path):
        """Load dataset from npz file.
        
        Args:
            data_path: Path to .npz file with 'inputs' and 'targets'
        """
        data = np.load(data_path)
        self.inputs = torch.from_numpy(data['inputs'])
        self.targets = torch.from_numpy(data['targets'])
        
        assert len(self.inputs) == len(self.targets), "Input/target mismatch"
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def train_epoch(
    model: FractalTransformer3D,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Returns:
        Tuple of (average loss, average max error)
    """
    model.train()
    total_loss = 0.0
    total_max_err = 0.0
    num_batches = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        with torch.no_grad():
            max_err = (outputs - targets).abs().max().item()
            total_max_err += max_err
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_max_err = total_max_err / num_batches
    
    return avg_loss, avg_max_err


def validate(
    model: FractalTransformer3D,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model.
    
    Returns:
        Tuple of (average loss, average max error)
    """
    model.eval()
    total_loss = 0.0
    total_max_err = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            max_err = (outputs - targets).abs().max().item()
            total_max_err += max_err
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_max_err = total_max_err / num_batches
    
    return avg_loss, avg_max_err


def train(
    model: FractalTransformer3D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_dir: Path,
) -> None:
    """Train the model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("-" * 80)
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_max_err = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_max_err = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Time: {epoch_time:5.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MaxErr: {val_max_err:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
    
    print("-" * 80)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer for fractal approximation")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "training_data.npz",
        help="Path to training data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Grid size",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Patch size",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on",
    )
    
    args = parser.parse_args()
    
    # Check data exists
    if not args.data.exists():
        print(f"Error: Training data not found: {args.data}")
        print("Please run generate_training_data.py first.")
        return
    
    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("Training Fractal Transformer")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading data from {args.data}...")
    full_dataset = FractalDataset(args.data)
    print(f"Total samples: {len(full_dataset)}")
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(
        grid_size=args.grid_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = model.to(device)
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

