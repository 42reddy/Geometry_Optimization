"""
Download and explore the AirfRANS dataset (NeurIPS 2022):
2D airfoil CFD simulations with point cloud velocity and pressure fields.
Uses PyTorch Geometric's efficient loader. Edit CONFIG section below.
"""

from pathlib import Path

import numpy as np

# ==== CONFIG: edit these variables to change behavior ====
OUTPUT_DIR = Path("data/airfrans")           # where to save dataset
N_SAMPLES = None                             # how many samples to keep (None = all ~1000)
SAVE_PREVIEW = True                          # whether to save preview PNG grids
TASK = "full"                                # task type: 'full', 'scarce', 'reynolds', 'aoa'
# =========================================================


def load_dataset_pyg(root: Path, task: str, train: bool, limit: int = None) -> list:
    """Load AirfRANS dataset using PyTorch Geometric's efficient loader."""
    try:
        from torch_geometric.datasets import AirfRANS
    except ImportError:
        raise ImportError(
            "PyTorch Geometric required. Install with:\n"
            "  pip install torch torch_geometric"
        )

    root.mkdir(parents=True, exist_ok=True)
    print(f"\nLoading AirfRANS {task} task, {'train' if train else 'test'} split...")

    dataset = AirfRANS(root=str(root), task=task, train=train, transform=None)

    if limit is not None:
        dataset = dataset[:limit]

    print(f"✓ Loaded {len(dataset)} samples")
    return dataset


def extract_fields(sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract coordinates, velocity magnitude, and pressure from a PyG sample.

    PyG Data structure:
    - pos: node coordinates (N, 2) - (x, y)
    - y: target fields (N, 4) - [v_x, v_y, pressure, nu_t]
    """
    # Handle both tensor and numpy formats
    pos = sample.pos
    if hasattr(pos, 'numpy'):
        pos = pos.numpy()
    else:
        pos = np.array(pos)

    targets = sample.y
    if hasattr(targets, 'numpy'):
        targets = targets.numpy()
    else:
        targets = np.array(targets)

    v_x = targets[:, 0].astype(np.float32)
    v_y = targets[:, 1].astype(np.float32)
    pressure = targets[:, 2].astype(np.float32)

    # Keep 2D coordinates as-is (x, y only)
    coords_2d = pos.astype(np.float32)

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(v_x**2 + v_y**2).astype(np.float32)

    return coords_2d, velocity_magnitude, pressure


def preview_pressure(dataset, sample_indices: list, out_png: Path) -> None:
    """Create 3x3 grid visualization of pressure fields (2D)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for grid_idx, sample_idx in enumerate(sample_indices[:9]):
        ax = axes[grid_idx]
        coords, vel_mag, pressure = extract_fields(dataset[sample_idx])

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=pressure, cmap="coolwarm", s=2, alpha=0.7
        )

        ax.set_aspect("equal")
        ax.set_title(f"Sample {sample_idx:04d}", fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, label="Pa", shrink=0.8)

    fig.suptitle("AirfRANS — Pressure Field (Pa)", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"✓ Saved pressure preview to {out_png}")
    plt.close()


def preview_velocity(dataset, sample_indices: list, out_png: Path) -> None:
    """Create 3x3 grid visualization of velocity magnitude fields (2D)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for grid_idx, sample_idx in enumerate(sample_indices[:9]):
        ax = axes[grid_idx]
        coords, vel_mag, pressure = extract_fields(dataset[sample_idx])

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=vel_mag, cmap="viridis", s=2, alpha=0.7
        )

        ax.set_aspect("equal")
        ax.set_title(f"Sample {sample_idx:04d}", fontsize=10)
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(sc, ax=ax, label="m/s", shrink=0.8)

    fig.suptitle("AirfRANS — Velocity Magnitude (m/s)", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"✓ Saved velocity preview to {out_png}")
    plt.close()


def print_sample_info(sample, sample_idx: int) -> None:
    """Print detailed information about a sample."""
    coords, vel_mag, pressure = extract_fields(sample)

    print(f"\nSample {sample_idx:04d} Information:")
    print(f"  Points: {coords.shape[0]} nodes")
    print(f"  Coordinate X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  Coordinate Y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  Velocity magnitude: [{vel_mag.min():.2f}, {vel_mag.max():.2f}] m/s")
    print(f"  Pressure: [{pressure.min():.1f}, {pressure.max():.1f}] Pa")

    # Print input features if available
    if hasattr(sample, 'x') and sample.x is not None:
        input_features = sample.x
        if hasattr(input_features, 'numpy'):
            input_features = input_features.numpy()
        else:
            input_features = np.array(input_features)

        v_x_free = input_features[0, 0] if input_features.shape[1] > 0 else 0
        v_y_free = input_features[0, 1] if input_features.shape[1] > 1 else 0
        print(f"  Freestream velocity: ({v_x_free:.2f}, {v_y_free:.2f}) m/s")


def main() -> None:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Load training data using PyTorch Geometric
    train_dataset = load_dataset_pyg(OUTPUT_DIR, task=TASK, train=True, limit=N_SAMPLES)

    # Print info about first sample
    print_sample_info(train_dataset[0], 0)

    # Create preview grids
    if SAVE_PREVIEW:
        sample_indices = list(range(min(9, len(train_dataset))))
        preview_pressure(train_dataset, sample_indices, OUTPUT_DIR / "preview_pressure.png")
        preview_velocity(train_dataset, sample_indices, OUTPUT_DIR / "preview_velocity.png")
        print(f"\n✓ Preview grids saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
