import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import igl
import numpy as np
import torch

# Model/config constants (mirrors the 2D image setup).
EMBEDDING_DIM = 512
ATTENTION_DIM = 256
AFFINE_HIDDEN_DIM = 128
MAX_CONTRAST = 1.8
SOFTMAX_TEMPERATURE = 0.25

# Grid/block configuration.
GRID_SIZE = 32
RANGE_SIZE = 4
DOMAIN_SIZE = 8
NUM_RANGE_BLOCKS = (GRID_SIZE // RANGE_SIZE) ** 3  # 8^3 = 512
NUM_DOMAIN_BLOCKS = (GRID_SIZE // DOMAIN_SIZE) ** 3  # 4^3 = 64

# Paths/seeds.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = SCRIPT_DIR / "spot.obj"
DEFAULT_SDF_CACHE = SCRIPT_DIR / "spot_sdf.npy"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "fractal_attention_sdf.pt"
DEFAULT_NUM_EPOCHS = 1200
DEFAULT_LR = 1e-3
SEED = 1337


def select_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def mesh_to_unit_bounds(V: np.ndarray, bound_half_extent: float = 0.5) -> np.ndarray:
    mesh_min = V.min(axis=0)
    mesh_max = V.max(axis=0)
    mesh_center = 0.5 * (mesh_min + mesh_max)
    mesh_extent = mesh_max - mesh_min
    max_extent = mesh_extent.max()
    if max_extent == 0:
        return V

    # Scale to live comfortably inside [-bound_half_extent, bound_half_extent]^3.
    scale = (2 * bound_half_extent * 0.9) / max_extent
    return (V - mesh_center) * scale


def sample_grid_points(dims: Tuple[int, int, int], bound_low, bound_high) -> np.ndarray:
    axes = [np.linspace(bound_low[i], bound_high[i], dims[i]) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 3)


def compute_sdf(mesh_path: Path, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        sdf = np.load(cache_path)
        if sdf.shape == (GRID_SIZE, GRID_SIZE, GRID_SIZE):
            print(f"Loaded cached SDF from {cache_path}")
            return sdf.astype(np.float32)
        else:
            print("Cached SDF has unexpected shape; recomputing.")

    bound_low = (-0.5, -0.5, -0.5)
    bound_high = (0.5, 0.5, 0.5)
    dims = (GRID_SIZE, GRID_SIZE, GRID_SIZE)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    V, F = igl.read_triangle_mesh(str(mesh_path))
    V = mesh_to_unit_bounds(V, bound_half_extent=bound_high[0])

    sample_pts = sample_grid_points(dims, bound_low, bound_high)
    distances, _, _, _ = igl.signed_distance(
        sample_pts,
        V,
        F,
        igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
    )
    sdf = distances.reshape(dims).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, sdf)
    print(f"Saved SDF grid to {cache_path}")
    return sdf


def get_blocks_3d(grid: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks_per_side = grid.shape[0] // block_size
    return (
        grid.view(blocks_per_side, block_size, blocks_per_side, block_size, blocks_per_side, block_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(blocks_per_side ** 3, block_size, block_size, block_size)
    )


def get_pooled_blocks_3d(grid: torch.Tensor, block_size: int) -> torch.Tensor:
    pooled = (
        grid[::2, ::2, ::2]
        + grid[::2, ::2, 1::2]
        + grid[::2, 1::2, ::2]
        + grid[::2, 1::2, 1::2]
        + grid[1::2, ::2, ::2]
        + grid[1::2, ::2, 1::2]
        + grid[1::2, 1::2, ::2]
        + grid[1::2, 1::2, 1::2]
    ) * 0.125
    return get_blocks_3d(pooled, block_size)


def reassemble_blocks_3d(blocks: torch.Tensor, grid_size: int) -> torch.Tensor:
    num_blocks, block_size, _, _ = blocks.shape
    blocks_per_side = grid_size // block_size
    return (
        blocks.view(blocks_per_side, blocks_per_side, blocks_per_side, block_size, block_size, block_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(grid_size, grid_size, grid_size)
    )


class FractalAttention3D(torch.nn.Module):
    def __init__(self, range_positional: torch.Tensor, domain_positional: torch.Tensor) -> None:
        super().__init__()
        self.register_parameter(
            "range_latents",
            torch.nn.Parameter(torch.randn(NUM_RANGE_BLOCKS, EMBEDDING_DIM)),
        )
        self.register_parameter(
            "domain_latents",
            torch.nn.Parameter(torch.randn(NUM_DOMAIN_BLOCKS, EMBEDDING_DIM)),
        )
        self.register_buffer("range_positional", range_positional)
        self.register_buffer("domain_positional", domain_positional)
        self.range_proj = torch.nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)
        self.domain_proj = torch.nn.Linear(EMBEDDING_DIM, ATTENTION_DIM)
        self.range_contrast_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)
        self.domain_contrast_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)
        self.range_offset_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)
        self.domain_offset_proj = torch.nn.Linear(EMBEDDING_DIM, AFFINE_HIDDEN_DIM)

    def forward(self, pooled_domains: torch.Tensor) -> torch.Tensor:
        range_base = self.range_latents + self.range_positional
        domain_base = self.domain_latents + self.domain_positional

        range_repr = self.range_proj(range_base)
        domain_repr = self.domain_proj(domain_base)

        range_contrast = self.range_contrast_proj(range_base)
        domain_contrast = self.domain_contrast_proj(domain_base)
        contrast_logits = (range_contrast @ domain_contrast.t()) / AFFINE_HIDDEN_DIM**0.5
        contrast = torch.tanh(contrast_logits) * MAX_CONTRAST

        range_offset = self.range_offset_proj(range_base)
        domain_offset = self.domain_offset_proj(domain_base)
        offset_logits = (range_offset @ domain_offset.t()) / AFFINE_HIDDEN_DIM**0.5
        offset = torch.tanh(offset_logits)

        logits = (range_repr @ domain_repr.t()) / (range_repr.shape[-1] ** 0.5 * SOFTMAX_TEMPERATURE)
        weights = torch.softmax(logits, dim=-1)

        domain_flat = pooled_domains.view(NUM_DOMAIN_BLOCKS, -1)
        scaled_domains = torch.matmul(weights * contrast, domain_flat)
        offset_effective = (weights * offset).sum(dim=1, keepdim=True)
        decoded = (scaled_domains + offset_effective).view(-1, RANGE_SIZE, RANGE_SIZE, RANGE_SIZE)
        return decoded


def collage_once(model: FractalAttention3D, source_grid: torch.Tensor) -> torch.Tensor:
    pooled = get_pooled_blocks_3d(source_grid, RANGE_SIZE)
    blocks = model(pooled)
    return reassemble_blocks_3d(blocks, GRID_SIZE)


def fmt(value: float) -> str:
    return f"{value:.4e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural PIFS on a mesh SDF.")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH,
        help="Path to OBJ mesh to convert to SDF (default: spot.obj).",
    )
    parser.add_argument(
        "--sdf-cache",
        type=Path,
        default=DEFAULT_SDF_CACHE,
        help="Where to cache the computed SDF grid (default: spot_sdf.npy).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Where to save the trained model (default: fractal_attention_sdf.pt).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU even if GPU is available.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    device = select_device(prefer_gpu=not args.cpu)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    elif device.type == "mps":
        torch.mps.manual_seed(SEED)  # type: ignore[attr-defined]

    sdf_np = compute_sdf(args.mesh, args.sdf_cache)
    sdf = torch.from_numpy(sdf_np).to(device)

    range_blocks = get_blocks_3d(sdf, RANGE_SIZE)
    pooled_domains = get_pooled_blocks_3d(sdf, RANGE_SIZE)
    range_positional = torch.zeros(NUM_RANGE_BLOCKS, EMBEDDING_DIM, device=device)
    domain_positional = torch.zeros(NUM_DOMAIN_BLOCKS, EMBEDDING_DIM, device=device)

    model = FractalAttention3D(range_positional, domain_positional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        start = time.perf_counter()
        pred_blocks = model(pooled_domains)
        loss = (pred_blocks - range_blocks).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()  # type: ignore[attr-defined]

        duration = time.perf_counter() - start
        print(f"Epoch {epoch + 1}/{args.num_epochs} - block MSE: {fmt(loss.item())} - {duration:.3f}s")

    model.eval()

    def evaluate_once(source_grid: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return collage_once(model, source_grid)

    with torch.no_grad():
        final_blocks = model(pooled_domains)
        per_block_mse = (final_blocks - range_blocks).pow(2).mean(dim=(1, 2, 3))
        mean_error = float(per_block_mse.mean().item())
        median_error = float(per_block_mse.median().item())

        rng = torch.Generator(device=device).manual_seed(42)
        iter_grid = torch.empty((GRID_SIZE, GRID_SIZE, GRID_SIZE), device=device)
        iter_grid.uniform_(-1.0, 1.0, generator=rng)
        for _ in range(16):
            iter_grid = evaluate_once(iter_grid)

        iter_mse = float((iter_grid - sdf).pow(2).mean().item())

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)

    print(f"Mean block MSE: {fmt(mean_error)}")
    print(f"Median block MSE: {fmt(median_error)}")
    print(f"MSE after 16 iterations vs SDF: {fmt(iter_mse)}")
    print(f"Saved checkpoint to {args.checkpoint}")


def load_model(
    checkpoint_path: Path,
    map_location: Optional[Union[torch.device, str]] = None,
) -> FractalAttention3D:
    range_positional = torch.zeros(NUM_RANGE_BLOCKS, EMBEDDING_DIM, device=map_location or "cpu")
    domain_positional = torch.zeros(NUM_DOMAIN_BLOCKS, EMBEDDING_DIM, device=map_location or "cpu")
    model = FractalAttention3D(range_positional, domain_positional)
    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
