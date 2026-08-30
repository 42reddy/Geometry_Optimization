"""
Evaluate trained FlowFieldPipeline on AirfRANS validation set.

Metrics from NeurIPS AirfRANS competition:
    Cl (lift coefficient):  ∫ p_normal_y / |airfoil| ds
    Cd (drag coefficient):  ∫ p_normal_x / |airfoil| ds
    Cp_rms (pressure RMS):  √(mean((p_pred - p_targ)²))

These are computed at boundary points (|SDF| < 0.5) where the physics
is most critical for aircraft design.
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train import AirfRANSGATrDataset, set_seed, SEED, CHECKPOINT_DIR, DATA_DIR, GRID_BOUNDS
from pipeline import FlowFieldPipeline

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = CHECKPOINT_DIR / "best.pt"
N_EVAL_SAMPLES = 10  # number of samples to evaluate and visualize
BOUNDARY_SDF_THRESHOLD = 0.5  # points with |SDF| < this are on/near boundary


def load_model(checkpoint_path: Path) -> nn.Module:
    """Load best checkpoint."""
    model = FlowFieldPipeline(
        input_scalar_dim=5,
        mv_channels=4,
        scalar_channels=8,
        n_heads=2,
        n_encoder_layers=4,
        grid_resolution=(64, 128),
        grid_bounds=GRID_BOUNDS,
        knn_k=16,
        bipartite_k=8,
        fno_hidden_channels=32,
        fno_layers=4,
        fno_modes=16,
        n_outputs=3,
    ).to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def compute_aerodynamic_metrics(
    coords: np.ndarray,
    sdf: np.ndarray,
    normals: np.ndarray,
    velocity_pred: np.ndarray,
    velocity_targ: np.ndarray,
    pressure_pred: np.ndarray,
    pressure_targ: np.ndarray,
    v_inf_mag: float,
) -> dict:
    """
    Compute lift, drag, and pressure RMS error at boundary points.

    AirfRANS uses chord-normalized coordinates (chord=1), and our pressure
    is already nondimensional (p / |V_inf|^2), which is the pressure coefficient.

    Force coefficients integrate surface pressure weighted by normal vectors:
        Cl = ∫ Cp * n_y ds  (lift = pressure integrated in y-direction)
        Cd = ∫ Cp * n_x ds  (drag = pressure integrated in x-direction)

    We approximate this sum by averaging Cp * n over boundary points,
    scaled by a nominal airfoil chord of 1.0.
    """

    # Identify boundary points: |SDF| < threshold means close to airfoil
    boundary_mask = np.abs(sdf) < BOUNDARY_SDF_THRESHOLD

    if not np.any(boundary_mask):
        # No boundary points found — return zeros
        return {
            'Cl': 0.0, 'Cd': 0.0,
            'Cl_error': 0.0, 'Cd_error': 0.0,
            'Cp_rms': 0.0, 'v_rms': 0.0,
            'n_boundary': 0
        }

    # Extract boundary quantities
    n_b = normals[boundary_mask]  # (N_boundary, 2)
    p_pred_b = pressure_pred[boundary_mask]  # (N_boundary,)
    p_targ_b = pressure_targ[boundary_mask]  # (N_boundary,)

    # Compute force coefficients (approximate by averaging Cp * n over boundary)
    # For a 2D airfoil with chord = 1, integrating becomes averaging over points
    Cl_pred = np.mean(p_pred_b * n_b[:, 1])  # y-component of pressure force
    Cl_targ = np.mean(p_targ_b * n_b[:, 1])

    Cd_pred = np.mean(p_pred_b * n_b[:, 0])  # x-component of pressure force
    Cd_targ = np.mean(p_targ_b * n_b[:, 0])

    # Errors
    Cl_error = np.abs(Cl_pred - Cl_targ)
    Cd_error = np.abs(Cd_pred - Cd_targ)

    # Pressure RMS at boundary (most critical for aerodynamics)
    Cp_rms = np.sqrt(np.mean((p_pred_b - p_targ_b) ** 2))

    # Velocity RMS everywhere
    v_rms = np.sqrt(np.mean((velocity_pred - velocity_targ) ** 2))

    return {
        'Cl_pred': Cl_pred, 'Cl_targ': Cl_targ, 'Cl_error': Cl_error,
        'Cd_pred': Cd_pred, 'Cd_targ': Cd_targ, 'Cd_error': Cd_error,
        'Cp_rms': Cp_rms, 'v_rms': v_rms,
        'n_boundary': np.sum(boundary_mask)
    }


def plot_sample_predictions(
    coords: np.ndarray,
    sdf: np.ndarray,
    velocity_pred: np.ndarray,
    velocity_targ: np.ndarray,
    pressure_pred: np.ndarray,
    pressure_targ: np.ndarray,
    metrics: dict,
    sample_idx: int,
    output_dir: Path = Path("eval_plots")
) -> None:
    """Visualize predicted vs target fields for one sample."""
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx} — Predicted vs Target\n"
                 f"Cl_pred={metrics['Cl_pred']:.4f} (target={metrics['Cl_targ']:.4f}), "
                 f"Cd_pred={metrics['Cd_pred']:.4f} (target={metrics['Cd_targ']:.4f})",
                 fontsize=12)

    # Velocity magnitude
    v_mag_pred = np.linalg.norm(velocity_pred, axis=1)
    v_mag_targ = np.linalg.norm(velocity_targ, axis=1)

    sc = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=v_mag_pred, cmap='viridis', s=10)
    axes[0, 0].set_title("Predicted Velocity Magnitude")
    axes[0, 0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 0], label='|V|')

    sc = axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=v_mag_targ, cmap='viridis', s=10)
    axes[0, 1].set_title("Target Velocity Magnitude")
    axes[0, 1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 1], label='|V|')

    v_mag_error = np.abs(v_mag_pred - v_mag_targ)
    sc = axes[0, 2].scatter(coords[:, 0], coords[:, 1], c=v_mag_error, cmap='hot', s=10)
    axes[0, 2].set_title(f"Velocity Error (RMS={metrics['v_rms']:.5f})")
    axes[0, 2].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 2], label='|Error|')

    # Pressure
    sc = axes[1, 0].scatter(coords[:, 0], coords[:, 1], c=pressure_pred, cmap='RdBu_r', s=10)
    axes[1, 0].set_title("Predicted Pressure (Cp)")
    axes[1, 0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 0], label='Cp')

    sc = axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=pressure_targ, cmap='RdBu_r', s=10)
    axes[1, 1].set_title("Target Pressure (Cp)")
    axes[1, 1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 1], label='Cp')

    p_error = np.abs(pressure_pred - pressure_targ)
    sc = axes[1, 2].scatter(coords[:, 0], coords[:, 1], c=p_error, cmap='hot', s=10)
    axes[1, 2].set_title(f"Pressure Error (RMS={metrics['Cp_rms']:.5f})")
    axes[1, 2].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 2], label='|Error|')

    plt.tight_layout()
    plt.savefig(output_dir / f"sample_{sample_idx:05d}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_dir / f'sample_{sample_idx:05d}.png'}")


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}\n")

    # Load data and model
    dataset = AirfRANSGATrDataset(DATA_DIR)
    print(f"Loaded {len(dataset)} samples from {DATA_DIR}\n")

    model = load_model(MODEL_PATH)

    # Create a small validation loader (first N_EVAL_SAMPLES)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluation metrics aggregator
    all_metrics = {
        'Cl_error': [], 'Cd_error': [],
        'Cl_pred': [], 'Cl_targ': [],
        'Cd_pred': [], 'Cd_targ': [],
        'Cp_rms': [], 'v_rms': []
    }

    print(f"Evaluating on {min(N_EVAL_SAMPLES, len(dataset))} samples...\n")

    with torch.no_grad():
        for sample_idx, (coords, node_scalars, target) in enumerate(loader):
            if sample_idx >= N_EVAL_SAMPLES:
                break

            coords = coords.squeeze(0).to(DEVICE)
            node_scalars = node_scalars.squeeze(0).to(DEVICE)
            target = target.squeeze(0).to(DEVICE)

            # Prediction
            pred = model(coords, node_scalars, coords)  # (N, 3): v_x, v_y, pressure

            # Convert to numpy for metrics
            coords_np = coords.cpu().numpy()
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            # Extract components
            velocity_pred = pred_np[:, :2]  # (N, 2)
            velocity_targ = target_np[:, :2]
            pressure_pred = pred_np[:, 2]  # (N,)
            pressure_targ = target_np[:, 2]

            # SDF from input features (node_scalars = [freestream (2), sdf (1), normals (2)])
            # sdf is at index 2 in node_scalars
            sdf = node_scalars[:, 2].cpu().numpy()
            normals = node_scalars[:, 3:5].cpu().numpy()

            # Load v_inf_mag from stats file
            stats = np.load(DATA_DIR / f"sample_{sample_idx:05d}" / "stats.npz")
            v_inf_mag = float(stats['v_inf_mag'])

            # Compute metrics
            metrics = compute_aerodynamic_metrics(
                coords_np, sdf, normals,
                velocity_pred, velocity_targ,
                pressure_pred, pressure_targ,
                v_inf_mag
            )

            # Aggregate
            all_metrics['Cl_error'].append(metrics['Cl_error'])
            all_metrics['Cd_error'].append(metrics['Cd_error'])
            all_metrics['Cl_pred'].append(metrics['Cl_pred'])
            all_metrics['Cl_targ'].append(metrics['Cl_targ'])
            all_metrics['Cd_pred'].append(metrics['Cd_pred'])
            all_metrics['Cd_targ'].append(metrics['Cd_targ'])
            all_metrics['Cp_rms'].append(metrics['Cp_rms'])
            all_metrics['v_rms'].append(metrics['v_rms'])

            # Plot
            plot_sample_predictions(
                coords_np, sdf,
                velocity_pred, velocity_targ,
                pressure_pred, pressure_targ,
                metrics, sample_idx
            )

            print(f"Sample {sample_idx:3d}: "
                  f"Cl={metrics['Cl_pred']:7.4f} (targ={metrics['Cl_targ']:7.4f}, err={metrics['Cl_error']:.4f}) | "
                  f"Cd={metrics['Cd_pred']:7.4f} (targ={metrics['Cd_targ']:7.4f}, err={metrics['Cd_error']:.4f}) | "
                  f"Cp_rms={metrics['Cp_rms']:.5f} | v_rms={metrics['v_rms']:.5f}")

    # Summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (AirfRANS Competition Metrics)")
    print("="*80)
    print(f"Samples evaluated: {len(all_metrics['Cl_error'])}")
    print(f"\nLift Coefficient (Cl):")
    print(f"  Prediction error:  {np.mean(all_metrics['Cl_error']):.6f} ± {np.std(all_metrics['Cl_error']):.6f}")
    print(f"  Mean Cl (pred):    {np.mean(all_metrics['Cl_pred']):.6f}")
    print(f"  Mean Cl (target):  {np.mean(all_metrics['Cl_targ']):.6f}")
    print(f"\nDrag Coefficient (Cd):")
    print(f"  Prediction error:  {np.mean(all_metrics['Cd_error']):.6f} ± {np.std(all_metrics['Cd_error']):.6f}")
    print(f"  Mean Cd (pred):    {np.mean(all_metrics['Cd_pred']):.6f}")
    print(f"  Mean Cd (target):  {np.mean(all_metrics['Cd_targ']):.6f}")
    print(f"\nPressure Coefficient (Cp) RMS at boundary:")
    print(f"  Mean RMS error:    {np.mean(all_metrics['Cp_rms']):.6f} ± {np.std(all_metrics['Cp_rms']):.6f}")
    print(f"\nVelocity RMS (everywhere):")
    print(f"  Mean RMS error:    {np.mean(all_metrics['v_rms']):.6f} ± {np.std(all_metrics['v_rms']):.6f}")
    print("="*80)
    print(f"\nPlots saved to eval_plots/")


if __name__ == "__main__":
    main()
