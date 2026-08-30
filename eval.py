"""
Evaluate trained FlowFieldPipeline on the held-out AirfRANS validation split
(the exact same split train.py produced, reconstructed here with the same
SEED — evaluating on training samples would give falsely optimistic metrics).

Cl and Cd are dimensionless aerodynamic coefficients, so they are computed
from the model's native normalized pressure output (pressure / |V_inf|^2 is
already the pressure-coefficient scaling) — this is correct as-is and is
NOT rescaled.

Velocity and pressure RMS errors, and everything plotted, ARE rescaled back
to physical units (m/s, and kinematic pressure m^2/s^2) using each sample's
v_inf_mag from stats.npz, before computing any error. Comparing normalized
errors across samples is meaningless because different samples have
different |V_inf|; physical units are also what let you sanity-check the
fields against real flow physics.

Metrics (mirroring NeurIPS AirfRANS competition):
    Cl (lift coefficient):  ~ ∫ Cp * n_y ds  (nondimensional)
    Cd (drag coefficient):  ~ ∫ Cp * n_x ds  (nondimensional)
    velocity RMS error:     physical units (m/s)
    pressure RMS error:     physical units (m^2/s^2, kinematic pressure)

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

from train import AirfRANSGATrDataset, set_seed, SEED, VAL_FRACTION, CHECKPOINT_DIR, DATA_DIR, GRID_BOUNDS
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
    velocity_pred_norm: np.ndarray,
    velocity_targ_norm: np.ndarray,
    pressure_pred_norm: np.ndarray,
    pressure_targ_norm: np.ndarray,
    velocity_pred_phys: np.ndarray,
    velocity_targ_phys: np.ndarray,
    pressure_pred_phys: np.ndarray,
    pressure_targ_phys: np.ndarray,
) -> dict:
    """
    Compute lift/drag coefficients (dimensionless, from normalized fields)
    and RMS errors (physical units, from rescaled fields) at boundary points.

    AirfRANS uses chord-normalized coordinates (chord=1). Our normalized
    pressure (p / |V_inf|^2) is already the pressure-coefficient scaling, so
    Cl/Cd MUST be computed from the *normalized* pressure — rescaling to
    physical units would make them depend on each sample's |V_inf|, which
    defeats the point of a dimensionless coefficient.

    Force coefficients integrate surface pressure weighted by normal vectors:
        Cl = ∫ Cp * n_y ds  (lift = pressure integrated in y-direction)
        Cd = ∫ Cp * n_x ds  (drag = pressure integrated in x-direction)

    We approximate this sum by averaging Cp * n over boundary points (the
    boundary points were farthest-point-sampled, so roughly uniform in arc
    length, making unweighted averaging a reasonable proxy for the true
    integral). Note this also omits the viscous (wall shear stress)
    contribution to Cd — AirfRANS ground-truth Cd includes it, so this is a
    pressure-drag-only approximation.

    RMS errors are computed from the physical-unit (rescaled) fields, since
    they're meant to answer "how far off are we in real terms" and different
    samples have different |V_inf|, making normalized-space errors not
    comparable across samples.
    """

    # Identify boundary points: |SDF| < threshold means close to airfoil
    boundary_mask = np.abs(sdf) < BOUNDARY_SDF_THRESHOLD

    if not np.any(boundary_mask):
        # No boundary points found — return zeros
        return {
            'Cl_pred': 0.0, 'Cl_targ': 0.0, 'Cl_error': 0.0,
            'Cd_pred': 0.0, 'Cd_targ': 0.0, 'Cd_error': 0.0,
            'p_rms': 0.0, 'v_rms': 0.0,
            'n_boundary': 0
        }

    # Extract boundary quantities (normalized, for Cl/Cd)
    n_b = normals[boundary_mask]  # (N_boundary, 2)
    p_pred_b = pressure_pred_norm[boundary_mask]  # (N_boundary,)
    p_targ_b = pressure_targ_norm[boundary_mask]  # (N_boundary,)

    # Compute force coefficients (approximate by averaging Cp * n over boundary)
    # For a 2D airfoil with chord = 1, integrating becomes averaging over points
    Cl_pred = np.mean(p_pred_b * n_b[:, 1])  # y-component of pressure force
    Cl_targ = np.mean(p_targ_b * n_b[:, 1])

    Cd_pred = np.mean(p_pred_b * n_b[:, 0])  # x-component of pressure force
    Cd_targ = np.mean(p_targ_b * n_b[:, 0])

    # Errors
    Cl_error = np.abs(Cl_pred - Cl_targ)
    Cd_error = np.abs(Cd_pred - Cd_targ)

    # Pressure RMS at boundary, physical units (kinematic pressure, m^2/s^2)
    p_pred_phys_b = pressure_pred_phys[boundary_mask]
    p_targ_phys_b = pressure_targ_phys[boundary_mask]
    p_rms = np.sqrt(np.mean((p_pred_phys_b - p_targ_phys_b) ** 2))

    # Velocity RMS everywhere, physical units (m/s)
    v_rms = np.sqrt(np.mean((velocity_pred_phys - velocity_targ_phys) ** 2))

    return {
        'Cl_pred': Cl_pred, 'Cl_targ': Cl_targ, 'Cl_error': Cl_error,
        'Cd_pred': Cd_pred, 'Cd_targ': Cd_targ, 'Cd_error': Cd_error,
        'p_rms': p_rms, 'v_rms': v_rms,
        'n_boundary': np.sum(boundary_mask)
    }


def plot_sample_predictions(
    coords: np.ndarray,
    sdf: np.ndarray,
    velocity_pred_phys: np.ndarray,
    velocity_targ_phys: np.ndarray,
    pressure_pred_phys: np.ndarray,
    pressure_targ_phys: np.ndarray,
    metrics: dict,
    sample_idx: int,
    v_inf_mag: float,
    output_dir: Path = Path("eval_plots")
) -> None:
    """Visualize predicted vs target fields (physical units) for one sample."""
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx} — Predicted vs Target (physical units, |V_inf|={v_inf_mag:.2f} m/s)\n"
                 f"Cl_pred={metrics['Cl_pred']:.4f} (target={metrics['Cl_targ']:.4f}), "
                 f"Cd_pred={metrics['Cd_pred']:.4f} (target={metrics['Cd_targ']:.4f})",
                 fontsize=12)

    # Velocity magnitude (m/s)
    v_mag_pred = np.linalg.norm(velocity_pred_phys, axis=1)
    v_mag_targ = np.linalg.norm(velocity_targ_phys, axis=1)

    sc = axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=v_mag_pred, cmap='viridis', s=10)
    axes[0, 0].set_title("Predicted Velocity Magnitude (m/s)")
    axes[0, 0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 0], label='|V| (m/s)')

    sc = axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=v_mag_targ, cmap='viridis', s=10)
    axes[0, 1].set_title("Target Velocity Magnitude (m/s)")
    axes[0, 1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 1], label='|V| (m/s)')

    v_mag_error = np.abs(v_mag_pred - v_mag_targ)
    sc = axes[0, 2].scatter(coords[:, 0], coords[:, 1], c=v_mag_error, cmap='hot', s=10)
    axes[0, 2].set_title(f"Velocity Error (RMS={metrics['v_rms']:.4f} m/s)")
    axes[0, 2].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 2], label='|Error| (m/s)')

    # Pressure (kinematic, m^2/s^2)
    sc = axes[1, 0].scatter(coords[:, 0], coords[:, 1], c=pressure_pred_phys, cmap='RdBu_r', s=10)
    axes[1, 0].set_title("Predicted Pressure (m²/s²)")
    axes[1, 0].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 0], label='p/ρ (m²/s²)')

    sc = axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=pressure_targ_phys, cmap='RdBu_r', s=10)
    axes[1, 1].set_title("Target Pressure (m²/s²)")
    axes[1, 1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 1], label='p/ρ (m²/s²)')

    p_error = np.abs(pressure_pred_phys - pressure_targ_phys)
    sc = axes[1, 2].scatter(coords[:, 0], coords[:, 1], c=p_error, cmap='hot', s=10)
    axes[1, 2].set_title(f"Pressure Error (RMS={metrics['p_rms']:.4f} m²/s²)")
    axes[1, 2].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 2], label='|Error| (m²/s²)')

    plt.tight_layout()
    plt.savefig(output_dir / f"sample_{sample_idx:05d}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_dir / f'sample_{sample_idx:05d}.png'}")


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}\n")

    # Load full dataset, then reconstruct train.py's exact val split (same
    # SEED, same random_split call) so we only evaluate on samples the model
    # never trained on.
    dataset = AirfRANSGATrDataset(DATA_DIR)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    _, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Loaded {len(dataset)} total samples from {DATA_DIR} — "
          f"evaluating on {len(val_set)} held-out validation samples\n")

    model = load_model(MODEL_PATH)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Evaluation metrics aggregator
    all_metrics = {
        'Cl_error': [], 'Cd_error': [],
        'Cl_pred': [], 'Cl_targ': [],
        'Cd_pred': [], 'Cd_targ': [],
        'p_rms': [], 'v_rms': []
    }

    n_eval = min(N_EVAL_SAMPLES, len(val_set))
    print(f"Evaluating on {n_eval} samples...\n")

    with torch.no_grad():
        for loop_i, (coords, node_scalars, target) in enumerate(val_loader):
            if loop_i >= N_EVAL_SAMPLES:
                break

            # val_set is a Subset — recover the real dataset index so we load
            # the correct sample's stats.npz (its directory, not loop_i).
            global_idx = val_set.indices[loop_i]
            sample_dir = dataset.sample_dirs[global_idx]

            coords = coords.squeeze(0).to(DEVICE)
            node_scalars = node_scalars.squeeze(0).to(DEVICE)
            target = target.squeeze(0).to(DEVICE)

            # Prediction — model outputs normalized (nondimensional) fields,
            # same space it was trained on.
            pred = model(coords, node_scalars, coords)  # (N, 3): v_x, v_y, pressure

            coords_np = coords.cpu().numpy()
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()

            velocity_pred_norm = pred_np[:, :2]  # (N, 2)
            velocity_targ_norm = target_np[:, :2]
            pressure_pred_norm = pred_np[:, 2]  # (N,)
            pressure_targ_norm = target_np[:, 2]

            # SDF from input features (node_scalars = [freestream (2), sdf (1), normals (2)])
            sdf = node_scalars[:, 2].cpu().numpy()
            normals = node_scalars[:, 3:5].cpu().numpy()

            # Rescale to physical units using this sample's v_inf_mag:
            #   velocity_phys = velocity_norm * |V_inf|
            #   pressure_phys = pressure_norm * |V_inf|^2
            stats = np.load(sample_dir / "stats.npz")
            v_inf_mag = float(stats['v_inf_mag'])

            velocity_pred_phys = velocity_pred_norm * v_inf_mag
            velocity_targ_phys = velocity_targ_norm * v_inf_mag
            pressure_pred_phys = pressure_pred_norm * (v_inf_mag ** 2)
            pressure_targ_phys = pressure_targ_norm * (v_inf_mag ** 2)

            metrics = compute_aerodynamic_metrics(
                coords_np, sdf, normals,
                velocity_pred_norm, velocity_targ_norm,
                pressure_pred_norm, pressure_targ_norm,
                velocity_pred_phys, velocity_targ_phys,
                pressure_pred_phys, pressure_targ_phys,
            )

            all_metrics['Cl_error'].append(metrics['Cl_error'])
            all_metrics['Cd_error'].append(metrics['Cd_error'])
            all_metrics['Cl_pred'].append(metrics['Cl_pred'])
            all_metrics['Cl_targ'].append(metrics['Cl_targ'])
            all_metrics['Cd_pred'].append(metrics['Cd_pred'])
            all_metrics['Cd_targ'].append(metrics['Cd_targ'])
            all_metrics['p_rms'].append(metrics['p_rms'])
            all_metrics['v_rms'].append(metrics['v_rms'])

            plot_sample_predictions(
                coords_np, sdf,
                velocity_pred_phys, velocity_targ_phys,
                pressure_pred_phys, pressure_targ_phys,
                metrics, global_idx, v_inf_mag
            )

            print(f"Sample {global_idx:5d} (|V_inf|={v_inf_mag:6.2f} m/s): "
                  f"Cl={metrics['Cl_pred']:7.4f} (targ={metrics['Cl_targ']:7.4f}, err={metrics['Cl_error']:.4f}) | "
                  f"Cd={metrics['Cd_pred']:7.4f} (targ={metrics['Cd_targ']:7.4f}, err={metrics['Cd_error']:.4f}) | "
                  f"p_rms={metrics['p_rms']:.4f} m²/s² | v_rms={metrics['v_rms']:.4f} m/s")

    # Summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY (held-out validation split, AirfRANS-style metrics)")
    print("="*80)
    print(f"Samples evaluated: {len(all_metrics['Cl_error'])}")
    print(f"\nLift Coefficient (Cl) — dimensionless:")
    print(f"  Prediction error:  {np.mean(all_metrics['Cl_error']):.6f} ± {np.std(all_metrics['Cl_error']):.6f}")
    print(f"  Mean Cl (pred):    {np.mean(all_metrics['Cl_pred']):.6f}")
    print(f"  Mean Cl (target):  {np.mean(all_metrics['Cl_targ']):.6f}")
    print(f"\nDrag Coefficient (Cd) — dimensionless, pressure-drag only (no viscous term):")
    print(f"  Prediction error:  {np.mean(all_metrics['Cd_error']):.6f} ± {np.std(all_metrics['Cd_error']):.6f}")
    print(f"  Mean Cd (pred):    {np.mean(all_metrics['Cd_pred']):.6f}")
    print(f"  Mean Cd (target):  {np.mean(all_metrics['Cd_targ']):.6f}")
    print(f"\nPressure RMS error at boundary — physical units (m²/s², kinematic):")
    print(f"  Mean RMS error:    {np.mean(all_metrics['p_rms']):.6f} ± {np.std(all_metrics['p_rms']):.6f}")
    print(f"\nVelocity RMS error everywhere — physical units (m/s):")
    print(f"  Mean RMS error:    {np.mean(all_metrics['v_rms']):.6f} ± {np.std(all_metrics['v_rms']):.6f}")
    print("="*80)
    print(f"\nPlots saved to eval_plots/")


if __name__ == "__main__":
    main()
