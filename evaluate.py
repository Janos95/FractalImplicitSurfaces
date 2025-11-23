"""Evaluate the trained transformer model against classical fractal code."""
import argparse
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from model import create_model
from vis import apply_partitioned_ifs


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model hyperparameters from checkpoint or use defaults
    # (Note: In production, you'd want to save these in the checkpoint)
    model = create_model(
        grid_size=32,
        patch_size=4,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Best val loss: {checkpoint['val_loss']:.6f}")
    
    return model


def load_fractal_code(mapping_path: Path) -> Dict[str, Any]:
    """Load fractal code from file.
    
    Args:
        mapping_path: Path to fractal code .npz file
        
    Returns:
        Fractal code mapping dictionary
    """
    mapping_data = np.load(mapping_path)
    mapping: Dict[str, Any] = {
        "range_block_size": int(mapping_data["range_block_size"]),
        "domain_block_size": int(mapping_data["domain_block_size"]),
        "best_domain_idx": mapping_data["best_domain_idx"],
        "best_sym_idx": mapping_data["best_sym_idx"],
        "scale": mapping_data["scale"],
        "offset": mapping_data["offset"],
    }
    return mapping


def evaluate(
    model: torch.nn.Module,
    fractal_mapping: Dict[str, Any],
    num_test_samples: int,
    grid_dims: tuple,
    device: torch.device,
    seed: int = 123,
) -> None:
    """Evaluate model against classical fractal code.
    
    Args:
        model: Trained transformer model
        fractal_mapping: Classical fractal code mapping
        num_test_samples: Number of test samples to generate
        grid_dims: Grid dimensions
        device: Device to run on
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    
    print(f"\nEvaluating on {num_test_samples} test samples...")
    print(f"Grid dimensions: {grid_dims}")
    print("-" * 80)
    
    # Metrics
    transformer_mse_list = []
    transformer_max_err_list = []
    transformer_times = []
    fractal_times = []
    
    with torch.no_grad():
        for i in range(num_test_samples):
            # Generate random input grid
            random_grid = rng.uniform(-1.0, 1.0, size=grid_dims).astype(np.float32)
            
            # Apply classical fractal code
            t0 = time.perf_counter()
            fractal_output = apply_partitioned_ifs(random_grid, fractal_mapping)
            fractal_time = time.perf_counter() - t0
            fractal_times.append(fractal_time)
            
            # Apply transformer model
            input_tensor = torch.from_numpy(random_grid).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            transformer_output = model(input_tensor)
            transformer_time = time.perf_counter() - t0
            transformer_times.append(transformer_time)
            
            transformer_output_np = transformer_output.squeeze(0).cpu().numpy()
            
            # Compute metrics (transformer vs fractal code)
            diff = transformer_output_np - fractal_output
            mse = np.mean(diff ** 2)
            max_err = np.max(np.abs(diff))
            
            transformer_mse_list.append(mse)
            transformer_max_err_list.append(max_err)
            
            if (i + 1) % 10 == 0:
                print(f"  Sample {i+1:3d}/{num_test_samples}: "
                      f"MSE={mse:.6f}, MaxErr={max_err:.6f}, "
                      f"Time: Fractal={fractal_time*1000:.2f}ms, "
                      f"Transformer={transformer_time*1000:.2f}ms")
    
    # Compute statistics
    mean_mse = np.mean(transformer_mse_list)
    std_mse = np.std(transformer_mse_list)
    mean_max_err = np.mean(transformer_max_err_list)
    std_max_err = np.std(transformer_max_err_list)
    
    mean_fractal_time = np.mean(fractal_times) * 1000  # ms
    mean_transformer_time = np.mean(transformer_times) * 1000  # ms
    
    print("-" * 80)
    print("EVALUATION RESULTS")
    print("-" * 80)
    print(f"Transformer vs Fractal Code:")
    print(f"  MSE:           {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"  Max Error:     {mean_max_err:.6f} ± {std_max_err:.6f}")
    print(f"\nInference Time:")
    print(f"  Fractal Code:  {mean_fractal_time:.2f} ms")
    print(f"  Transformer:   {mean_transformer_time:.2f} ms")
    print(f"  Speedup:       {mean_fractal_time / mean_transformer_time:.2f}x")
    print()
    
    # Compute compression metrics
    # Fractal code size
    fractal_size = (
        fractal_mapping['best_domain_idx'].nbytes +
        fractal_mapping['best_sym_idx'].nbytes +
        fractal_mapping['scale'].nbytes +
        fractal_mapping['offset'].nbytes +
        16  # metadata
    )
    
    # Model size (parameters)
    model_params = sum(p.numel() for p in model.parameters())
    model_size = model_params * 4  # float32
    
    # Grid size
    grid_size = np.prod(grid_dims) * 4  # float32
    
    print("COMPRESSION METRICS")
    print("-" * 80)
    print(f"Grid size:         {grid_size / 1024:.2f} KB")
    print(f"Fractal code size: {fractal_size / 1024:.2f} KB")
    print(f"Model size:        {model_size / 1024 / 1024:.2f} MB ({model_params:,} params)")
    print(f"Fractal compression ratio: {grid_size / fractal_size:.2f}x")
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transformer model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--fractal-code",
        type=Path,
        default=Path(__file__).resolve().parent / "fractal_code.npz",
        help="Path to fractal code",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test samples",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Grid size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train the model first using train.py")
        return
    
    if not args.fractal_code.exists():
        print(f"Error: Fractal code not found: {args.fractal_code}")
        print("Please run vis.py and click 'Compress' to generate the fractal code.")
        return
    
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
    print("Evaluating Fractal Transformer")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model...")
    model = load_model(args.checkpoint, device)
    
    # Load fractal code
    print(f"\nLoading fractal code...")
    fractal_mapping = load_fractal_code(args.fractal_code)
    print(f"  Range block size: {fractal_mapping['range_block_size']}")
    print(f"  Domain block size: {fractal_mapping['domain_block_size']}")
    
    # Evaluate
    grid_dims = (args.grid_size, args.grid_size, args.grid_size)
    evaluate(
        model=model,
        fractal_mapping=fractal_mapping,
        num_test_samples=args.num_samples,
        grid_dims=grid_dims,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

