import argparse
from itertools import permutations, product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import time

# 24 proper rotations (no reflections) of the cube as axis permutations with sign flips.
def _generate_cube_rotations() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    rots: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for perm in permutations((0, 1, 2)):
        for flips in product((-1, 1), repeat=3):
            mat = np.zeros((3, 3), dtype=int)
            for i, p in enumerate(perm):
                mat[i, p] = flips[i]
            if round(np.linalg.det(mat)) == 1:
                rots.append((perm, flips))
    return rots


SYMMETRIES: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = _generate_cube_rotations()


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


def sample_grid_points(dims: tuple[int, int, int], bound_low, bound_high) -> np.ndarray:
    axes = [
        np.linspace(bound_low[i], bound_high[i], dims[i]) for i in range(3)
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack(grid, axis=-1).reshape(-1, 3)


def apply_symmetry(block: np.ndarray, perm: Sequence[int], flips: Sequence[int]) -> np.ndarray:
    rotated = np.transpose(block, axes=perm)
    for axis, flip in enumerate(flips):
        if flip == -1:
            rotated = np.flip(rotated, axis=axis)
    return rotated


def extract_blocks(field: np.ndarray, block_size: int) -> np.ndarray:
    blocks_per_dim = field.shape[0] // block_size
    reshaped = field.reshape(
        blocks_per_dim,
        block_size,
        blocks_per_dim,
        block_size,
        blocks_per_dim,
        block_size,
    )
    reordered = reshaped.transpose(0, 2, 4, 1, 3, 5)
    return reordered.reshape(-1, block_size, block_size, block_size)


def downsample_block(block: np.ndarray, factor: int) -> np.ndarray:
    if block.shape[0] % factor != 0:
        raise ValueError("Block size must be divisible by factor")
    new_size = block.shape[0] // factor
    reshaped = block.reshape(
        new_size,
        factor,
        new_size,
        factor,
        new_size,
        factor,
    )
    return reshaped.mean(axis=(1, 3, 5))


def downsample_blocks(blocks: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return blocks
    if blocks.shape[1] % factor != 0:
        raise ValueError("Block size must be divisible by factor")
    new_size = blocks.shape[1] // factor
    reshaped = blocks.reshape(
        blocks.shape[0],
        new_size,
        factor,
        new_size,
        factor,
        new_size,
        factor,
    )
    return reshaped.mean(axis=(2, 4, 6))


def compute_partitioned_ifs(
    source_grid: np.ndarray,
    range_block_size: int = 4,
    domain_block_size: int = 8,
) -> Dict[str, Any]:
    range_blocks = extract_blocks(source_grid, range_block_size)
    domain_blocks = extract_blocks(source_grid, domain_block_size)
    factor = domain_block_size // range_block_size

    # Precompute all symmetric variants of downsampled domain blocks.
    n_domain = domain_blocks.shape[0]
    n_sym = len(SYMMETRIES)
    domain_variants = np.empty(
        (n_domain, n_sym, range_block_size, range_block_size, range_block_size),
        dtype=source_grid.dtype,
    )
    for di, block in enumerate(domain_blocks):
        for si, (perm, flips) in enumerate(SYMMETRIES):
            rotated = apply_symmetry(block, perm, flips)
            domain_variants[di, si] = downsample_block(rotated, factor)

    n_range = range_blocks.shape[0]
    samples_per_block = range_block_size ** 3

    rng_flat = range_blocks.reshape(n_range, samples_per_block)
    rng_mean = rng_flat.mean(axis=1)
    rng_centered = rng_flat - rng_mean[:, None]
    rng_var = (rng_centered ** 2).mean(axis=1)

    dom_flat = domain_variants.reshape(-1, samples_per_block)
    dom_mean = dom_flat.mean(axis=1)
    dom_centered = dom_flat - dom_mean[:, None]
    dom_var = (dom_centered ** 2).mean(axis=1)

    n_candidates = dom_flat.shape[0]
    best_error = np.full(n_range, np.inf)
    best_idx = np.zeros(n_range, dtype=int)
    best_cov = np.zeros(n_range)
    best_dom_var = np.zeros(n_range)
    best_dom_mean = np.zeros(n_range)

    chunk = 256
    for start in range(0, n_candidates, chunk):
        end = min(start + chunk, n_candidates)
        dom_chunk = dom_flat[start:end]
        dom_mean_chunk = dom_mean[start:end]
        dom_var_chunk = dom_var[start:end]
        dom_centered_chunk = dom_chunk - dom_mean_chunk[:, None]

        cov = dom_centered_chunk @ rng_centered.T / samples_per_block
        safe_dom_var = np.where(dom_var_chunk < 1e-12, np.inf, dom_var_chunk)
        error = rng_var[None, :] - (cov ** 2) / safe_dom_var[:, None]

        chunk_min = error.min(axis=0)
        chunk_arg = error.argmin(axis=0)
        update_mask = chunk_min < best_error
        if not np.any(update_mask):
            continue

        update_indices = np.nonzero(update_mask)[0]
        best_error[update_indices] = chunk_min[update_indices]
        chosen = chunk_arg[update_indices]
        best_idx[update_indices] = chosen + start
        best_cov[update_indices] = cov[chosen, update_indices]
        best_dom_var[update_indices] = safe_dom_var[chosen]
        best_dom_mean[update_indices] = dom_mean_chunk[chosen]

    best_scale = best_cov / best_dom_var
    best_offset = rng_mean - best_scale * best_dom_mean

    best_domain_idx = best_idx // n_sym
    best_sym_idx = best_idx % n_sym

    return {
        "range_block_size": range_block_size,
        "domain_block_size": domain_block_size,
        "best_domain_idx": best_domain_idx,
        "best_sym_idx": best_sym_idx,
        "scale": best_scale,
        "offset": best_offset,
    }


def apply_partitioned_ifs(
    source_grid: np.ndarray,
    mapping: Dict[str, Any],
) -> np.ndarray:
    rbs = mapping["range_block_size"]
    dbs = mapping["domain_block_size"]
    n_sym = len(SYMMETRIES)
    blocks_per_dim = source_grid.shape[0] // rbs

    domain_blocks = extract_blocks(source_grid, dbs)
    factor = dbs // rbs

    result = np.empty_like(source_grid)
    idx = 0
    for bz in range(blocks_per_dim):
        for by in range(blocks_per_dim):
            for bx in range(blocks_per_dim):
                dom_idx = mapping["best_domain_idx"][idx]
                sym_idx = mapping["best_sym_idx"][idx]
                perm, flips = SYMMETRIES[sym_idx % n_sym]
                block = apply_symmetry(domain_blocks[dom_idx], perm, flips)
                block = downsample_block(block, factor)
                mapped = mapping["scale"][idx] * block + mapping["offset"][idx]

                z0 = bz * rbs
                y0 = by * rbs
                x0 = bx * rbs
                result[z0 : z0 + rbs, y0 : y0 + rbs, x0 : x0 + rbs] = mapped
                idx += 1

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polyscope IFS demo")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(__file__).resolve().parent / "spot.obj",
        help="Path to mesh to use for SDF (default: spot.obj next to vis.py)",
    )
    return parser.parse_args()


def main(mesh_path: Path) -> None:
    dims = (32, 32, 32)
    bound_low = (-0.5, -0.5, -0.5)
    bound_high = (0.5, 0.5, 0.5)

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
    sdf = distances.reshape(dims)

    iter_grid = np.random.uniform(-1.0, 1.0, size=dims)
    state: Dict[str, Any] = {
        "mapping": None,
        "iter_grid": iter_grid,
        "last_compress_ms": None,
        "last_mse": None,
        "last_max_err": None,
    }

    ps.init()
    ps_grid = ps.register_volume_grid("spot sdf", dims, bound_low, bound_high)
    ps_iter_grid = ps.register_volume_grid("iter grid", dims, bound_low, bound_high)

    ps_grid.add_scalar_quantity(
        "spot signed distance",
        sdf,
        defined_on="nodes",
        enable_isosurface_viz=True,
        isosurface_level=0.0,
        enable_gridcube_viz=False,
        isosurface_color=(0.2, 0.4, 0.8),
    )

    def update_iter_visual(enabled: bool = True) -> None:
        ps_iter_grid.add_scalar_quantity(
            "iter values",
            state["iter_grid"],
            defined_on="nodes",
            enable_isosurface_viz=True,
            isosurface_level=0.0,
            enable_gridcube_viz=False,
            isosurface_color=(0.8, 0.3, 0.2),
            enabled=enabled,
        )

    update_iter_visual(enabled=True)

    def ui_callback() -> None:
        if psim.Button("Compress"):
            start = time.perf_counter()
            state["mapping"] = compute_partitioned_ifs(
                sdf,
                range_block_size=4,
                domain_block_size=8,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            state["last_compress_ms"] = elapsed_ms
            print(f"Computed partitioned IFS mapping in {elapsed_ms:.2f} ms.")
            
            # Save fractal code to disk
            mapping = state["mapping"]
            save_path = Path(__file__).resolve().parent / "fractal_code.npz"
            np.savez_compressed(
                save_path,
                range_block_size=mapping["range_block_size"],
                domain_block_size=mapping["domain_block_size"],
                best_domain_idx=mapping["best_domain_idx"],
                best_sym_idx=mapping["best_sym_idx"],
                scale=mapping["scale"],
                offset=mapping["offset"],
            )
            print(f"Saved fractal code to {save_path}")

        if psim.Button("Iterate once"):
            if state["mapping"] is None:
                print("Please run Compress before iterating.")
            else:
                state["iter_grid"] = apply_partitioned_ifs(
                    state["iter_grid"], state["mapping"]
                )
                update_iter_visual(enabled=True)
                diff = state["iter_grid"] - sdf
                state["last_mse"] = float(np.mean(diff ** 2))
                state["last_max_err"] = float(np.max(np.abs(diff)))

        if psim.Button("Reset"):
            state["iter_grid"] = np.random.uniform(-1.0, 1.0, size=dims)
            update_iter_visual(enabled=True)
            state["last_mse"] = None
            state["last_max_err"] = None
            print("Reset iter grid.")

        if state["last_compress_ms"] is None:
            psim.Text("Last compress: (not run yet)")
        else:
            psim.Text(f"Last compress: {state['last_compress_ms']:.2f} ms")

        if state["last_mse"] is None:
            psim.Text("Last MSE to sdf: (not computed)")
        else:
            psim.Text(f"Last MSE to sdf: {state['last_mse']:.3e}")

        if state["last_max_err"] is None:
            psim.Text("Last max |err| to sdf: (not computed)")
        else:
            psim.Text(f"Last max |err| to sdf: {state['last_max_err']:.3e}")

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.mesh)
