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
VOXEL_PREFILTER_FACTOR = 3  # pre-filter each bin to ~factor*n_bin before FPS

# Log-spaced |sdf| bin edges (chord units) used to STRATIFY the downsample
# explicitly, replacing the old "80% from a loose |sdf|<0.5 boundary blob"
# heuristic. That heuristic ran farthest-point sampling over the whole
# boundary blob, which maximizes spatial COVERAGE, not density -- so it
# spread the budget roughly evenly across |sdf|<0.5 and ended up putting
# only ~2% of the 9k budget inside |sdf|<0.01, even though that thin
# viscous-sublayer shell holds ~47% of the true ~180k-point mesh (RANS
# solvers cluster cells there to resolve the boundary layer). Any point
# never seen at training density can't be learned, no matter how the loss
# is weighted. Binning geometrically (not linearly) matches how the mesh
# itself refines -- each bin roughly halves/doubles distance from the wall.
SDF_BIN_EDGES = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf'))
# Combined floor on the fraction of target_points spent inside |sdf|<0.01,
# applied ON TOP OF that band's own true-mesh fraction -- "match or exceed"
# the true near-wall density, since the steep gradients there benefit from
# more than a naive count-proportional share.
NEAR_WALL_FLOOR_FRACTION = 0.30
# Bins fully beyond this |sdf| are cheap uniform random instead of FPS --
# that part of the flow is smooth and doesn't need exact spatial coverage.
FPS_SDF_CUTOFF = 0.5
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


def voxel_prefilter(coords: np.ndarray, max_points: int, iters: int = 20) -> np.ndarray:
    """
    Fast vectorized pre-filter: bin points into a 2D grid sized to yield
    roughly `max_points` occupied cells, keep one point per cell.

    Cell size is found by binary search on the ACHIEVED occupied-cell count
    rather than a bounding-box-area formula (area / max_points). The area
    formula assumes points fill the 2D box uniformly; it badly under-counts
    when points are confined to a thin curve/manifold inside that box --
    e.g. points right at the airfoil surface live on a ~1D curve, so a
    handful of cells covering the curve's bounding box are almost all
    empty, and the naive formula returns far fewer occupied cells than
    requested (observed: ~700 instead of ~9800 for a near-wall bin). Search
    is data-driven, so it self-corrects regardless of whether the bin's
    points fill an area or hug a curve.

    This is O(n) per probe (no Python-level distance loop) and is used to
    shrink a large point set before running the O(n*k) farthest point
    sampling, so FPS only ever runs on a small, already-well-spread subset.
    """
    n = coords.shape[0]
    if n <= max_points:
        return np.arange(n)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)

    def occupied(cell_size: float) -> np.ndarray:
        grid_idx = np.floor((coords - mins) / cell_size).astype(np.int64)
        keys = grid_idx[:, 0] * 1_000_003 + grid_idx[:, 1]
        _, unique_pos = np.unique(keys, return_index=True)
        return unique_pos

    cell_lo = float(extent.min()) * 1e-6   # fine enough that occupied(cell_lo) ~= n
    cell_hi = float(extent.max())          # coarse enough that occupied(cell_hi) ~= 1

    best = occupied(cell_lo)
    if len(best) <= max_points:
        return best   # even the finest cell size can't reach max_points (near-duplicate coords)

    # Binary search for the LARGEST cell_size that still yields >= max_points
    # occupied cells (largest -> closest to max_points from above, keeping
    # the candidate set small for the FPS step that follows).
    lo, hi = cell_lo, cell_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        occ = occupied(mid)
        if len(occ) >= max_points:
            lo, best = mid, occ
        else:
            hi = mid
    return best


def compute_bin_budget(
    abs_sdf: np.ndarray,
    target_points: int,
    bin_edges=SDF_BIN_EDGES,
    near_wall_floor_fraction: float = NEAR_WALL_FLOOR_FRACTION,
):
    """
    Per-sample point budget for each |sdf| bin: proportional to that bin's
    TRUE share of the full mesh's points, then topped up so bins fully
    inside |sdf|<0.01 collectively get at least `near_wall_floor_fraction`
    of target_points (pulled proportionally from the other bins so the
    total still equals target_points exactly).
    """
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    n_bins = len(bin_edges) - 1
    n_total = max(len(abs_sdf), 1)

    bin_of_point = np.digitize(abs_sdf, bin_edges[1:-1], right=False)  # 0..n_bins-1
    true_counts = np.array([(bin_of_point == b).sum() for b in range(n_bins)], dtype=np.float64)
    true_fractions = true_counts / n_total

    budget = true_fractions * target_points

    near_wall_bins = [b for b in range(n_bins) if bin_edges[b + 1] <= 0.01 + 1e-9]
    floor_total = near_wall_floor_fraction * target_points
    near_wall_total = budget[near_wall_bins].sum() if near_wall_bins else 0.0

    if near_wall_bins and near_wall_total < floor_total:
        weights = true_fractions[near_wall_bins]
        weights = weights / weights.sum() if weights.sum() > 1e-9 else np.full(len(near_wall_bins), 1.0 / len(near_wall_bins))
        deficit = floor_total - near_wall_total

        other_bins = [b for b in range(n_bins) if b not in near_wall_bins]
        other_total = budget[other_bins].sum()
        if other_total > 1e-9:
            shrink = max(0.0, (other_total - deficit) / other_total)
            budget[other_bins] *= shrink

        for b, w in zip(near_wall_bins, weights):
            budget[b] += deficit * w

    bin_budget = np.round(budget).astype(int)
    drift = target_points - bin_budget.sum()
    if n_bins > 0:
        bin_budget[np.argmax(bin_budget)] += drift  # absorb rounding drift in the largest bin
    return np.clip(bin_budget, 0, None), bin_of_point


def sdf_stratified_downsample(
    coords: np.ndarray,
    sdf: np.ndarray,
    normals: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    target_points: int,
):
    """
    Downsample by explicit |sdf| stratification instead of a loose
    boundary/field split. Each bin's budget (see compute_bin_budget) matches
    or exceeds that bin's true share of the full mesh, so the near-wall
    viscous sublayer is no longer systematically thinned out relative to
    the real mesh's own refinement. Within each bin, points are chosen the
    same way the old boundary sampling did (voxel pre-filter + FPS) for
    bins near the wall, or plain uniform random for bins entirely in the
    smooth far field.
    """
    abs_sdf = np.abs(sdf)
    bin_budget, bin_of_point = compute_bin_budget(abs_sdf, target_points)
    bin_edges = np.asarray(SDF_BIN_EDGES, dtype=np.float64)

    final_indices = []
    for b in range(len(bin_budget)):
        n_b = int(bin_budget[b])
        if n_b <= 0:
            continue
        bin_indices = np.where(bin_of_point == b)[0]
        if len(bin_indices) == 0:
            continue
        if len(bin_indices) <= n_b:
            final_indices.append(bin_indices)
            continue

        if bin_edges[b] >= FPS_SDF_CUTOFF:
            chosen_local = np.random.choice(len(bin_indices), size=n_b, replace=False)
        else:
            prefilter_target = min(len(bin_indices), n_b * VOXEL_PREFILTER_FACTOR)
            prefiltered_local = voxel_prefilter(coords[bin_indices], prefilter_target)
            if len(prefiltered_local) > n_b:
                fps_local = farthest_point_sample(coords[bin_indices][prefiltered_local], n_b)
                chosen_local = prefiltered_local[fps_local]
            else:
                chosen_local = prefiltered_local
        final_indices.append(bin_indices[chosen_local])

    final_indices = np.concatenate(final_indices) if final_indices else np.arange(0)

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
    coords_full: np.ndarray,
    sdf_full: np.ndarray,
    normals_full: np.ndarray,
    velocity_full: np.ndarray,
    pressure_full: np.ndarray,
) -> None:
    """Save downsampled sample (for training) plus the full-resolution mesh
    (for evaluation) in NPZ format."""
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

    # Full ~180k-point mesh, nondimensionalized with the SAME v_inf_mag as the
    # downsampled data above (see prepare_dataset: normalization happens once,
    # before downsampling, so data.npz and full.npz can't drift apart in
    # scaling). GridDecoder can be queried at arbitrary coordinates, so at
    # eval time the model can be decoded onto these full-resolution points
    # even though it only ever encodes the downsampled point cloud — this is
    # what makes it possible to measure true full-mesh error instead of only
    # error on the (boundary-heavy, therefore harder) training point cloud.
    np.savez(
        sample_dir / "full.npz",
        coords=coords_full,
        sdf=sdf_full,
        normals=normals_full,
        velocity=velocity_full,
        pressure=pressure_full,
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
    print(f"SDF-stratified sampling: near-wall (|sdf|<0.01) floor = {NEAR_WALL_FLOOR_FRACTION:.0%} of budget")

    stats_all = []

    for idx in range(dataset_size):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{dataset_size}...", end='\r')

        # Load sample
        coords, sdf, normals, freestream, velocity, pressure = load_airfrans_sample(
            airfrans_dataset, idx
        )

        # Nondimensionalize the FULL-resolution fields first, then downsample
        # the already-normalized arrays for training — this guarantees
        # data.npz (downsampled) and full.npz (for eval) use identical
        # scaling instead of risking two separate nondimensionalize() calls
        # drifting apart.
        normalized_full, stats = nondimensionalize(freestream, velocity, pressure)

        # Downsample
        coords_down, sdf_down, normals_down, vel_down, press_down, _ = sdf_stratified_downsample(
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
            coords_full=coords,
            sdf_full=sdf,
            normals_full=normals,
            velocity_full=normalized_full['velocity'],
            pressure_full=normalized_full['pressure'],
        )

        # Track statistics (convert numpy types to Python native for JSON serialization)
        near_wall_frac_full = float((np.abs(sdf) < 0.01).mean())
        near_wall_frac_down = float((np.abs(sdf_down) < 0.01).mean())
        stats_all.append({
            'sample_id': int(idx),
            'n_points': int(len(coords)),
            'n_points_down': int(len(coords_down)),
            'boundary_ratio': float((np.abs(sdf_down) < 0.5).mean()),
            'near_wall_frac_full_mesh': near_wall_frac_full,
            'near_wall_frac_downsampled': near_wall_frac_down,
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
    print(f"  Avg boundary ratio (|sdf|<0.5): {np.mean([s['boundary_ratio'] for s in stats_all]):.1%}")
    print(f"  Near-wall (|sdf|<0.01) fraction — full mesh:   "
          f"{np.mean([s['near_wall_frac_full_mesh'] for s in stats_all]):.1%}")
    print(f"  Near-wall (|sdf|<0.01) fraction — downsampled: "
          f"{np.mean([s['near_wall_frac_downsampled'] for s in stats_all]):.1%}  "
          f"(should now be >= the full-mesh figure, not ~2% as before)")
    print(f"  Freestream speed range: [{min(s['v_inf_mag'] for s in stats_all):.2f}, "
          f"{max(s['v_inf_mag'] for s in stats_all):.2f}]")

    # Save metadata
    metadata = {
        'n_samples': dataset_size,
        'target_points': target_points,
        'sdf_bin_edges': list(SDF_BIN_EDGES),
        'near_wall_floor_fraction': NEAR_WALL_FLOOR_FRACTION,
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
