"""
Prepare AirfRANS data for GINOT encoding:
1. Extract the full-resolution airfoil surface contour (geometry encoder input)
2. Intelligently downsample the volume point cloud (180k → ~9k) as query/target points
3. Preserve boundary geometry (airfoil)
4. Nondimensionalize fields by freestream speed
5. Save in GINOT-compatible format
"""

from pathlib import Path
import numpy as np
from typing import Tuple

# ==== CONFIG ====
AIRFRANS_DIR = Path("data/airfrans")
OUTPUT_DIR = Path("data/airfrans_gatr")
N_SAMPLES = None  # None = all, or int for subset
TARGET_POINTS = 9000  # downsample target
VOXEL_PREFILTER_FACTOR = 3  # pre-filter each near-wall bin to ~factor*n_bin before FPS

# log-spaced |sdf| bin edges used to stratify the downsample so the budget
# tracks the true mesh's radial density instead of a flat "80% near
# boundary" split. Edges get finer the closer to the wall (down to 1e-4
# chord), since that's where AirfRANS itself concentrates points and where
# the flow gradients are sharpest (viscous sublayer).
SDF_BIN_EDGES = (0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, np.inf)
# Multiplicative boost applied to the true-mesh fraction of bins fully
# inside sdf<0.01 before renormalizing, so the downsample matches *or
# exceeds* (rather than just matches) the true viscous-sublayer density —
# undersampling here is far costlier than oversampling it, since that's
# where velocity/pressure gradients are steepest.
NEAR_WALL_BOOST = 1.3
NEAR_WALL_BOOST_CUTOFF = 0.01
# ================


def load_airfrans_sample(dataset, idx: int):
    """Load a single AirfRANS sample and extract features."""
    sample = dataset[idx]

    # Convert to numpy if needed
    pos = sample.pos.numpy() if hasattr(sample.pos, 'numpy') else np.array(sample.pos)
    x = sample.x.numpy() if hasattr(sample.x, 'numpy') else np.array(sample.x)
    y = sample.y.numpy() if hasattr(sample.y, 'numpy') else np.array(sample.y)

    # x: [V_x_free, V_y_free, SDF, normal_x, normal_y]  (V_x_free, V_y_free constant per sample)
    # y: [v_x, v_y, pressure, nu_t]

    coords = pos.astype(np.float32)  # (N, 2)
    sdf = x[:, 2].astype(np.float32)  # signed distance to airfoil, chord-normalized
    normals = x[:, 3:5].astype(np.float32)  # (N, 2) unit surface normals
    freestream = x[:, 0:2].mean(axis=0).astype(np.float32)  # (2,) constant per sample
    velocity = y[:, :2].astype(np.float32)  # (N, 2)
    pressure = y[:, 2].astype(np.float32)  # (N,)

    return coords, sdf, normals, freestream, velocity, pressure


def voxel_prefilter(coords: np.ndarray, max_points: int) -> np.ndarray:
    """
    Fast vectorized pre-filter: bin points into a 2D grid sized to yield
    roughly `max_points` occupied cells, keep one point per cell.

    This is O(n) (no Python-level distance loop) and is used to shrink a
    large point set before running the O(n*k) farthest point sampling, so
    FPS only ever runs on a small, already-well-spread subset.
    """
    n = coords.shape[0]
    if n <= max_points:
        return np.arange(n)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    area = extent[0] * extent[1]
    cell_size = np.sqrt(area / max_points)
    cell_size = max(cell_size, 1e-9)

    grid_idx = np.floor((coords - mins) / cell_size).astype(np.int64)
    # Combine (row, col) grid indices into a single hashable key per point
    keys = grid_idx[:, 0] * 1_000_003 + grid_idx[:, 1]
    _, unique_pos = np.unique(keys, return_index=True)
    return unique_pos


def compute_bin_budget(sdf: np.ndarray, target_points: int, bin_edges=SDF_BIN_EDGES) -> np.ndarray:
    """
    Allocate `target_points` across |sdf| bins so the downsample tracks the
    true mesh's radial density, boosted (not just matched) in the viscous
    sublayer (|sdf| < NEAR_WALL_BOOST_CUTOFF).

    Returns an integer array of per-bin point counts summing to
    target_points (subject to each bin's own point count, handled by the
    caller).
    """
    abs_sdf = np.abs(sdf)
    n_total = len(sdf)
    n_bins = len(bin_edges) - 1

    true_frac = np.empty(n_bins)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        true_frac[i] = np.sum((abs_sdf >= lo) & (abs_sdf < hi)) / n_total

    boosted_frac = true_frac.copy()
    for i in range(n_bins):
        if bin_edges[i + 1] <= NEAR_WALL_BOOST_CUTOFF:
            boosted_frac[i] *= NEAR_WALL_BOOST

    boosted_frac /= boosted_frac.sum()

    budget = np.floor(boosted_frac * target_points).astype(np.int64)
    # Distribute the rounding remainder to the largest-fraction bins so the
    # total still lands exactly on target_points.
    remainder = target_points - budget.sum()
    if remainder > 0:
        order = np.argsort(-boosted_frac)
        for i in order[:remainder]:
            budget[i] += 1

    return budget


def boundary_aware_downsample(
    coords: np.ndarray,
    sdf: np.ndarray,
    normals: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    target_points: int,
    bin_edges=SDF_BIN_EDGES,
):
    """
    Downsample by explicit |sdf| strata rather than a flat boundary/field
    split, so the budget always tracks (and, inside the viscous sublayer,
    exceeds) the true mesh's radial point density — see compute_bin_budget.

    Within each bin: voxel pre-filter + farthest point sampling for spatial
    coverage in the near-wall bins (where geometric structure matters and
    the true point count is usually small anyway), plain random sampling in
    the far-field bin (large, smooth, so exact coverage doesn't matter and
    FPS there would be needlessly expensive).
    """
    abs_sdf = np.abs(sdf)
    n_bins = len(bin_edges) - 1
    budget = compute_bin_budget(sdf, target_points, bin_edges)

    final_indices = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        n_bin = int(budget[i])
        if n_bin <= 0:
            continue

        bin_indices = np.where((abs_sdf >= lo) & (abs_sdf < hi))[0]
        if len(bin_indices) <= n_bin:
            final_indices.append(bin_indices)
            continue

        is_far_field = hi == np.inf
        if is_far_field:
            sampled_local = np.random.choice(len(bin_indices), size=n_bin, replace=False)
            final_indices.append(bin_indices[sampled_local])
        else:
            prefilter_target = min(len(bin_indices), n_bin * VOXEL_PREFILTER_FACTOR)
            prefiltered_local = voxel_prefilter(coords[bin_indices], prefilter_target)
            bin_candidates = bin_indices[prefiltered_local]

            if len(bin_candidates) > n_bin:
                sampled_local = farthest_point_sample(coords[bin_candidates], n_bin)
                final_indices.append(bin_candidates[sampled_local])
            else:
                final_indices.append(bin_candidates)

    # Any budget left unfilled by bins that ran out of points (rare, only
    # when a bin's true point count is below its allocated budget) is
    # backfilled from the largest remaining bin so we still hit target_points.
    final_indices = np.concatenate(final_indices)
    shortfall = target_points - len(final_indices)
    if shortfall > 0:
        remaining_mask = np.ones(len(sdf), dtype=bool)
        remaining_mask[final_indices] = False
        remaining_indices = np.where(remaining_mask)[0]
        if len(remaining_indices) > 0:
            extra = np.random.choice(
                remaining_indices, size=min(shortfall, len(remaining_indices)), replace=False
            )
            final_indices = np.concatenate([final_indices, extra])

    # Subsample all per-node arrays
    coords_down = coords[final_indices]
    sdf_down = sdf[final_indices]
    normals_down = normals[final_indices]
    velocity_down = velocity[final_indices]
    pressure_down = pressure[final_indices]

    return coords_down, sdf_down, normals_down, velocity_down, pressure_down, final_indices


def farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest point sampling to preserve geometric structure."""
    n = points.shape[0]
    if n <= n_samples:
        return np.arange(n)

    sampled_indices = [np.random.randint(n)]
    distances = np.full(n, np.inf)

    for _ in range(n_samples - 1):
        # Distance from newly sampled point to all others
        dists = np.linalg.norm(points - points[sampled_indices[-1]], axis=1)
        # Update minimum distance to any sampled point
        distances = np.minimum(distances, dists)
        # Pick point with max distance
        sampled_indices.append(np.argmax(distances))

    return np.array(sampled_indices)


def nondimensionalize(
    freestream: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray
) -> Tuple[dict, dict]:
    """
    Nondimensionalize by freestream speed |V_inf| — the standard CFD
    normalization, and physically grounded (unlike a per-sample z-score):
    it ties every sample to the same reference regardless of how fast or
    slow that particular simulation's inlet condition was.

    Coordinates, SDF, and normals are left untouched: AirfRANS already
    expresses them in a consistent chord-normalized frame (chord = 1)
    across every sample, so re-normalizing them per-sample would make the
    same airfoil geometry look different from one sample to the next.

    velocity_star = velocity / |V_inf|
    pressure_star = pressure / |V_inf|^2         (standard dynamic-pressure scaling)
    freestream_star = freestream / |V_inf|        (becomes the unit AoA direction vector)
    """
    v_inf_mag = float(np.linalg.norm(freestream))
    v_inf_mag = v_inf_mag if v_inf_mag > 1e-6 else 1.0

    normalized = {
        'velocity': velocity / v_inf_mag,
        'pressure': pressure / (v_inf_mag ** 2),
        'freestream': freestream / v_inf_mag,
    }
    stats = {'v_inf_mag': v_inf_mag}

    return normalized, stats


def save_sample(
    output_dir: Path,
    idx: int,
    coords: np.ndarray,
    sdf: np.ndarray,
    normals: np.ndarray,
    freestream: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    stats: dict,
    surface_coords: np.ndarray,
) -> None:
    """Save the downsampled sample in NPZ format. `surface_coords` is the
    full-resolution airfoil contour (GINOT's geometry encoder input); `sdf`
    stays only for physical diagnostics (e.g. eval.py's sdf-banded metrics),
    since GINOT needs no SDF/normals as model input."""
    sample_dir = output_dir / f"sample_{idx:05d}"
    sample_dir.mkdir(exist_ok=True)

    # log(|V_inf|) as an extra scalar condition feature: `freestream` above is
    # already divided by |V_inf|, so it only carries the AoA direction — the
    # flow *speed* (and therefore Reynolds number, since AirfRANS varies
    # viscosity much less than inlet speed) would otherwise never reach the
    # model, even though the nondimensional flow shape still depends on Re.
    # log-scale keeps the range small (AirfRANS speeds ~10-80 m/s -> ~2.3-4.4)
    # without needing dataset-wide normalization stats.
    log_v_inf = np.array(np.log(stats['v_inf_mag']), dtype=np.float32)

    # Save as NPZ (single binary file with multiple arrays)
    np.savez(
        sample_dir / "data.npz",
        coords=coords,
        sdf=sdf,
        normals=normals,
        freestream=freestream,
        velocity=velocity,
        pressure=pressure,
        log_v_inf=log_v_inf,
        surface_coords=surface_coords,
    )

    # Save stats as NPZ (needed to denormalize predictions back to physical units)
    np.savez(sample_dir / "stats.npz", **stats)


def prepare_dataset(
    airfrans_dataset,
    output_dir: Path,
    n_samples: int = None,
    target_points: int = 9000
) -> None:
    """Process entire dataset for GINOT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_size = len(airfrans_dataset)
    if n_samples is not None:
        dataset_size = min(n_samples, dataset_size)

    print(f"\nPreparing {dataset_size} samples for GINOT")
    print(f"Target: {target_points} points per sample")
    print(f"SDF bin edges: {SDF_BIN_EDGES}")

    stats_all = []

    for idx in range(dataset_size):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{dataset_size}...", end='\r')

        # Load sample
        coords, sdf, normals, freestream, velocity, pressure = load_airfrans_sample(
            airfrans_dataset, idx
        )
        # Full-resolution airfoil contour — GINOT's geometry encoder input.
        # Surface mesh nodes have sdf exactly 0 (they sit on the body).
        surface_coords = coords[sdf == 0]

        normalized_full, stats = nondimensionalize(freestream, velocity, pressure)

        # Downsample
        coords_down, sdf_down, normals_down, vel_down, press_down, _ = boundary_aware_downsample(
            coords, sdf, normals, normalized_full['velocity'], normalized_full['pressure'],
            target_points
        )

        # Save
        save_sample(
            output_dir, idx,
            coords_down,
            sdf_down,
            normals_down,
            normalized_full['freestream'],
            vel_down,
            press_down,
            stats,
            surface_coords,
        )

        # Track statistics (convert numpy types to Python native for JSON serialization)
        stats_all.append({
            'sample_id': int(idx),
            'n_points': int(len(coords)),
            'n_points_down': int(len(coords_down)),
            'boundary_ratio': float((np.abs(sdf_down) < 0.5).mean()),
            'near_wall_frac_down': float((np.abs(sdf_down) < NEAR_WALL_BOOST_CUTOFF).mean()),
            'near_wall_frac_true': float((np.abs(sdf) < NEAR_WALL_BOOST_CUTOFF).mean()),
            'v_inf_mag': stats['v_inf_mag'],
            'pressure_range': [float(pressure.min()), float(pressure.max())],
            'velocity_mag_range': [float(np.linalg.norm(velocity, axis=1).min()),
                                   float(np.linalg.norm(velocity, axis=1).max())]
        })

    print(f"\n✓ Processed {dataset_size} samples")

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Avg compression: {np.mean([s['n_points'] for s in stats_all]):.0f} → "
          f"{np.mean([s['n_points_down'] for s in stats_all]):.0f} points")
    print(f"  Avg boundary ratio: {np.mean([s['boundary_ratio'] for s in stats_all]):.1%}")
    print(f"  Avg near-wall (|sdf|<{NEAR_WALL_BOOST_CUTOFF}) fraction: "
          f"{np.mean([s['near_wall_frac_down'] for s in stats_all]):.1%} downsampled vs "
          f"{np.mean([s['near_wall_frac_true'] for s in stats_all]):.1%} true mesh")
    print(f"  Freestream speed range: [{min(s['v_inf_mag'] for s in stats_all):.2f}, "
          f"{max(s['v_inf_mag'] for s in stats_all):.2f}]")

    # Save metadata
    metadata = {
        'n_samples': dataset_size,
        'target_points': target_points,
        'sdf_bin_edges': [None if e == np.inf else e for e in SDF_BIN_EDGES],
        'near_wall_boost': NEAR_WALL_BOOST,
        'near_wall_boost_cutoff': NEAR_WALL_BOOST_CUTOFF,
        'sample_stats': stats_all
    }

    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    # Load AirfRANS dataset
    try:
        from torch_geometric.datasets import AirfRANS
    except ImportError:
        print("Error: PyTorch Geometric required. Install with: pip install torch_geometric")
        return

    print("Loading AirfRANS dataset...")
    dataset = AirfRANS(root=str(AIRFRANS_DIR), task="full", train=True)
    print(f"✓ Loaded {len(dataset)} samples")

    # Prepare for GINOT
    prepare_dataset(dataset, OUTPUT_DIR, N_SAMPLES, TARGET_POINTS)

    print(f"\n✓ Data saved to {OUTPUT_DIR}/")
    print(f"  Ready for GINOT encoding!")


if __name__ == "__main__":
    main()
