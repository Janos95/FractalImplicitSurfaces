import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
DEFAULT_GRID_SIZE = 32
RANGE_SIZE = 4
DOMAIN_SIZE = 8

# Paths/seeds.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = SCRIPT_DIR / "spot.obj"
DEFAULT_SDF_CACHE = SCRIPT_DIR / "spot_sdf.npy"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "spot_32.pt"
DEFAULT_NUM_EPOCHS = 1200
DEFAULT_LR = 1e-3
SEED = 1337

@dataclass(frozen=True)
class GridConfig:
    grid_size: int
    range_size: int = RANGE_SIZE
    domain_size: int = DOMAIN_SIZE

    def __post_init__(self) -> None:
        if self.domain_size != self.range_size * 2:
            raise ValueError("Domain block size must be twice the range block size for pooling to work.")
        if self.grid_size % self.range_size != 0:
            raise ValueError(f"Grid size {self.grid_size} must be divisible by range block size {self.range_size}.")
        if self.grid_size % self.domain_size != 0:
            raise ValueError(f"Grid size {self.grid_size} must be divisible by domain block size {self.domain_size}.")

    @property
    def num_range_blocks(self) -> int:
        return (self.grid_size // self.range_size) ** 3

    @property
    def num_domain_blocks(self) -> int:
        return (self.grid_size // self.domain_size) ** 3


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


def compute_sdf(mesh_path: Path, cache_path: Path, grid_size: int) -> np.ndarray:
    if cache_path.exists():
        sdf = np.load(cache_path)
        if sdf.shape == (grid_size, grid_size, grid_size):
            print(f"Loaded cached SDF from {cache_path}")
            return sdf.astype(np.float32)
        else:
            print("Cached SDF has unexpected shape; recomputing.")

    bound_low = (-0.5, -0.5, -0.5)
    bound_high = (0.5, 0.5, 0.5)
    dims = (grid_size, grid_size, grid_size)

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
    def __init__(
        self,
        config: GridConfig,
        range_positional: torch.Tensor,
        domain_positional: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        if range_positional.shape[0] != config.num_range_blocks:
            raise ValueError("Range positional embedding shape does not match grid configuration.")
        if domain_positional.shape[0] != config.num_domain_blocks:
            raise ValueError("Domain positional embedding shape does not match grid configuration.")
        self.register_parameter(
            "range_latents",
            torch.nn.Parameter(torch.randn(config.num_range_blocks, EMBEDDING_DIM)),
        )
        self.register_parameter(
            "domain_latents",
            torch.nn.Parameter(torch.randn(config.num_domain_blocks, EMBEDDING_DIM)),
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

        domain_flat = pooled_domains.view(self.config.num_domain_blocks, -1)
        scaled_domains = torch.matmul(weights * contrast, domain_flat)
        offset_effective = (weights * offset).sum(dim=1, keepdim=True)
        decoded = (scaled_domains + offset_effective).view(
            -1, self.config.range_size, self.config.range_size, self.config.range_size
        )
        return decoded


def collage_once(model: FractalAttention3D, source_grid: torch.Tensor) -> torch.Tensor:
    pooled = get_pooled_blocks_3d(source_grid, model.config.range_size)
    blocks = model(pooled)
    return reassemble_blocks_3d(blocks, model.config.grid_size)


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
        help="Where to save the trained model (default: spot_32.pt).",
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
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Resolution of the cubic SDF grid (must be divisible by 8).",
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

    config = GridConfig(grid_size=args.grid_size)

    sdf_np = compute_sdf(args.mesh, args.sdf_cache, config.grid_size)
    sdf = torch.from_numpy(sdf_np).to(device)

    range_blocks = get_blocks_3d(sdf, config.range_size)
    pooled_domains = get_pooled_blocks_3d(sdf, config.range_size)
    range_positional = torch.zeros(config.num_range_blocks, EMBEDDING_DIM, device=device)
    domain_positional = torch.zeros(config.num_domain_blocks, EMBEDDING_DIM, device=device)

    model = FractalAttention3D(config, range_positional, domain_positional).to(device)
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
        iter_grid = torch.empty((config.grid_size, config.grid_size, config.grid_size), device=device)
        iter_grid.uniform_(-1.0, 1.0, generator=rng)
        for _ in range(16):
            iter_grid = evaluate_once(iter_grid)

        iter_mse = float((iter_grid - sdf).pow(2).mean().item())

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "grid_size": config.grid_size,
            "mesh_path": str(args.mesh.resolve()),
        },
        args.checkpoint,
    )

    print(f"Mean block MSE: {fmt(mean_error)}")
    print(f"Median block MSE: {fmt(median_error)}")
    print(f"MSE after 16 iterations vs SDF: {fmt(iter_mse)}")
    print(f"Saved checkpoint to {args.checkpoint}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
