"""Generate training data for transformer model by applying fractal IFS to random grids."""
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
from vis import apply_partitioned_ifs, SYMMETRIES, extract_blocks, downsample_block, apply_symmetry


def generate_training_data(
    mapping_path: Path,
    num_samples: int,
    grid_dims: tuple[int, int, int],
    output_path: Path,
    seed: int = 42,
) -> None:
    """Generate training data by applying fractal code to random grids.
    
    Args:
        mapping_path: Path to the fractal code (.npz file)
        num_samples: Number of training samples to generate
        grid_dims: Dimensions of the grid (e.g., (32, 32, 32))
        output_path: Path to save the training data
        seed: Random seed for reproducibility
    """
    # Load the fractal code
    print(f"Loading fractal code from {mapping_path}...")
    mapping_data = np.load(mapping_path)
    mapping: Dict[str, Any] = {
        "range_block_size": int(mapping_data["range_block_size"]),
        "domain_block_size": int(mapping_data["domain_block_size"]),
        "best_domain_idx": mapping_data["best_domain_idx"],
        "best_sym_idx": mapping_data["best_sym_idx"],
        "scale": mapping_data["scale"],
        "offset": mapping_data["offset"],
    }
    print(f"  Range block size: {mapping['range_block_size']}")
    print(f"  Domain block size: {mapping['domain_block_size']}")
    print(f"  Number of range blocks: {len(mapping['best_domain_idx'])}")
    
    # Generate random grids and their IFS-transformed versions
    rng = np.random.RandomState(seed)
    print(f"\nGenerating {num_samples} training samples...")
    
    inputs = np.empty((num_samples, *grid_dims), dtype=np.float32)
    targets = np.empty((num_samples, *grid_dims), dtype=np.float32)
    
    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
        
        # Generate random input grid
        random_grid = rng.uniform(-1.0, 1.0, size=grid_dims).astype(np.float32)
        
        # Apply fractal IFS transformation
        transformed_grid = apply_partitioned_ifs(random_grid, mapping)
        
        inputs[i] = random_grid
        targets[i] = transformed_grid
    
    print(f"\nSaving training data to {output_path}...")
    np.savez_compressed(
        output_path,
        inputs=inputs,
        targets=targets,
        grid_dims=np.array(grid_dims),
    )
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\nDone!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data for transformer")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).resolve().parent / "fractal_code.npz",
        help="Path to fractal code file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=32,
        help="Grid dimension (will be NxNxN)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "training_data.npz",
        help="Output path for training data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    if not args.mapping.exists():
        print(f"Error: Fractal code file not found: {args.mapping}")
        print("Please run vis.py and click 'Compress' to generate the fractal code first.")
        return
    
    grid_dims = (args.grid_size, args.grid_size, args.grid_size)
    
    generate_training_data(
        mapping_path=args.mapping,
        num_samples=args.num_samples,
        grid_dims=grid_dims,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

