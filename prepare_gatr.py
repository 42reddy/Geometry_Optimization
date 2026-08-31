"""
Prepare AirfRANS data for GATr encoding:
1. Intelligently downsample point clouds (180k → ~9k)
2. Preserve boundary geometry (airfoil)
3. Nondimensionalize fields by freestream speed
4. Save in GATr-compatible format
"""

from pathlib import Path
import numpy as np
from typing import Tuple

# ==== CONFIG ====
AIRFRANS_DIR = Path("data/airfrans")
OUTPUT_DIR = Path("data/airfrans_gatr")
N_SAMPLES = None  # None = all, or int for subset
TARGET_POINTS = 9000  # downsample target
BOUNDARY_RATIO = 0.80  # 80% of samples from boundary, 20% from field
VOXEL_PREFILTER_FACTOR = 3  # pre-filter boundary set to ~factor*n_boundary before FPS
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


def boundary_aware_downsample(
    coords: np.ndarray,
    sdf: np.ndarray,
    normals: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    target_points: int,
    boundary_ratio: float = 0.80
):
    """
    Intelligently downsample while preserving boundary geometry.

    Strategy:
    - Keep more points near airfoil (high information density), selected
      with a voxel pre-filter + farthest point sampling for good spatial
      coverage of the geometry.
    - Sparse, plain random sampling in the freestream (flow is smooth and
      slowly varying there, so exact spatial coverage isn't important),
      which is far cheaper than FPS at these point counts.
    - Use SDF to identify boundary regions.
    """

    n_boundary = int(target_points * boundary_ratio)
    n_field = target_points - n_boundary

    # Identify boundary points: those very close to airfoil
    # |SDF| < 0.5 means close to surface (negative = inside, positive = outside)
    boundary_mask = np.abs(sdf) < 0.5
    field_mask = ~boundary_mask

    boundary_indices = np.where(boundary_mask)[0]
    field_indices = np.where(field_mask)[0]

    # Boundary sampling: voxel pre-filter to shrink the set, then FPS on the
    # much smaller remainder to preserve geometric structure.
    if len(boundary_indices) > n_boundary:
        prefilter_target = min(len(boundary_indices), n_boundary * VOXEL_PREFILTER_FACTOR)
        prefiltered_local = voxel_prefilter(coords[boundary_indices], prefilter_target)
        boundary_candidates = boundary_indices[prefiltered_local]

        if len(boundary_candidates) > n_boundary:
            boundary_sampled = farthest_point_sample(
                coords[boundary_candidates], n_boundary
            )
            boundary_idx_final = boundary_candidates[boundary_sampled]
        else:
            boundary_idx_final = boundary_candidates
    else:
        boundary_idx_final = boundary_indices
        n_field = target_points - len(boundary_idx_final)

    # Field sampling: plain uniform random sampling (cheap, and sufficient
    # since the freestream field is smooth and doesn't need FPS coverage).
    if len(field_indices) > n_field:
        field_idx_final = np.random.choice(field_indices, size=n_field, replace=False)
    else:
        field_idx_final = field_indices

    # Combine indices
    final_indices = np.concatenate([boundary_idx_final, field_idx_final])

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
    stats: dict
) -> None:
    """Save downsampled sample in NPZ format."""
    sample_dir = output_dir / f"sample_{idx:05d}"
    sample_dir.mkdir(exist_ok=True)

    # log(|V_inf|) as an extra broadcast scalar feature: `freestream` above is
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
    )

    # Save stats as NPZ (needed to denormalize predictions back to physical units)
    np.savez(sample_dir / "stats.npz", **stats)


def prepare_dataset(
    airfrans_dataset,
    output_dir: Path,
    n_samples: int = None,
    target_points: int = 9000
) -> None:
    """Process entire dataset for GATr."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_size = len(airfrans_dataset)
    if n_samples is not None:
        dataset_size = min(n_samples, dataset_size)

    print(f"\nPreparing {dataset_size} samples for GATr")
    print(f"Target: {target_points} points per sample")
    print(f"Boundary ratio: {BOUNDARY_RATIO:.1%}")

    stats_all = []

    for idx in range(dataset_size):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{dataset_size}...", end='\r')

        # Load sample
        coords, sdf, normals, freestream, velocity, pressure = load_airfrans_sample(
            airfrans_dataset, idx
        )

        # Downsample
        coords_down, sdf_down, normals_down, vel_down, press_down, _ = boundary_aware_downsample(
            coords, sdf, normals, velocity, pressure, target_points, BOUNDARY_RATIO
        )

        # Nondimensionalize by freestream speed
        normalized, stats = nondimensionalize(freestream, vel_down, press_down)

        # Save
        save_sample(
            output_dir, idx,
            coords_down,
            sdf_down,
            normals_down,
            normalized['freestream'],
            normalized['velocity'],
            normalized['pressure'],
            stats
        )

        # Track statistics (convert numpy types to Python native for JSON serialization)
        stats_all.append({
            'sample_id': int(idx),
            'n_points': int(len(coords)),
            'n_points_down': int(len(coords_down)),
            'boundary_ratio': float((np.abs(sdf_down) < 0.5).mean()),
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
    print(f"  Freestream speed range: [{min(s['v_inf_mag'] for s in stats_all):.2f}, "
          f"{max(s['v_inf_mag'] for s in stats_all):.2f}]")

    # Save metadata
    metadata = {
        'n_samples': dataset_size,
        'target_points': target_points,
        'boundary_ratio': BOUNDARY_RATIO,
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

    # Prepare for GATr
    prepare_dataset(dataset, OUTPUT_DIR, N_SAMPLES, TARGET_POINTS)

    print(f"\n✓ Data saved to {OUTPUT_DIR}/")
    print(f"  Ready for GATr encoding!")


if __name__ == "__main__":
    main()
