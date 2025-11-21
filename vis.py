from pathlib import Path
from typing import Any, Dict

import igl
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


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
    domain_down = downsample_blocks(
        domain_blocks, factor=domain_block_size // range_block_size
    )

    n_range = range_blocks.shape[0]
    n_domain = domain_down.shape[0]
    samples_per_block = range_block_size ** 3

    rng_flat = range_blocks.reshape(n_range, samples_per_block)
    dom_flat = domain_down.reshape(n_domain, samples_per_block)

    rng_mean = rng_flat.mean(axis=1)
    dom_mean = dom_flat.mean(axis=1)
    rng_centered = rng_flat - rng_mean[:, None]
    dom_centered = dom_flat - dom_mean[:, None]

    rng_var = (rng_centered ** 2).mean(axis=1)
    dom_var = (dom_centered ** 2).mean(axis=1)

    cov = np.einsum("in,jn->ij", dom_centered, rng_centered) / samples_per_block

    safe_dom_var = np.where(dom_var < 1e-12, np.inf, dom_var)
    scale = cov / safe_dom_var[:, None]
    offset = rng_mean[None, :] - scale * dom_mean[:, None]

    error = rng_var[None, :] - (cov ** 2) / safe_dom_var[:, None]
    best_domain_idx = error.argmin(axis=0)
    idx_range = np.arange(n_range)
    best_scale = scale[best_domain_idx, idx_range]
    best_offset = offset[best_domain_idx, idx_range]

    return {
        "range_block_size": range_block_size,
        "domain_block_size": domain_block_size,
        "best_domain_idx": best_domain_idx,
        "scale": best_scale,
        "offset": best_offset,
    }


def apply_partitioned_ifs(
    source_grid: np.ndarray,
    mapping: Dict[str, Any],
) -> np.ndarray:
    rbs = mapping["range_block_size"]
    dbs = mapping["domain_block_size"]
    blocks_per_dim = source_grid.shape[0] // rbs

    domain_blocks = extract_blocks(source_grid, dbs)
    downsampled = downsample_blocks(domain_blocks, factor=dbs // rbs)

    result = np.empty_like(source_grid)
    idx = 0
    for bz in range(blocks_per_dim):
        for by in range(blocks_per_dim):
            for bx in range(blocks_per_dim):
                dom_idx = mapping["best_domain_idx"][idx]
                block = downsampled[dom_idx]
                mapped = mapping["scale"][idx] * block + mapping["offset"][idx]

                z0 = bz * rbs
                y0 = by * rbs
                x0 = bx * rbs
                result[z0 : z0 + rbs, y0 : y0 + rbs, x0 : x0 + rbs] = mapped
                idx += 1

    return result


def main() -> None:
    dims = (32, 32, 32)
    bound_low = (-0.5, -0.5, -0.5)
    bound_high = (0.5, 0.5, 0.5)

    mesh_path = Path(__file__).resolve().parent / "spot.obj"
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
    state: Dict[str, Any] = {"mapping": None, "iter_grid": iter_grid}

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
            state["mapping"] = compute_partitioned_ifs(
                sdf,
                range_block_size=4,
                domain_block_size=8,
            )
            print("Computed partitioned IFS mapping.")

        if psim.Button("Iterate once"):
            if state["mapping"] is None:
                print("Please run Compress before iterating.")
            else:
                state["iter_grid"] = apply_partitioned_ifs(
                    state["iter_grid"], state["mapping"]
                )
                update_iter_visual(enabled=True)
                print("Applied mapping once.")

        if psim.Button("Reset"):
            state["iter_grid"] = np.random.uniform(-1.0, 1.0, size=dims)
            update_iter_visual(enabled=True)
            print("Reset iter grid.")

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == "__main__":
    main()
