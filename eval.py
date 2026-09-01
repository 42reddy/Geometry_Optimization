"""
Evaluate trained FlowFieldPipeline on the held-out AirfRANS validation split
(the exact same split train.py produced, reconstructed here with the same
SEED — evaluating on training samples would give falsely optimistic
metrics).

The model both ENCODES and is DECODED at the downsampled ~9k point cloud
from data.npz (prepare_gatr.py doesn't save a full-resolution mesh).


Two kinds of quantities are reported, and they are NOT interchangeable:

  - Cl, Cd (lift/drag coefficients) are dimensionless. They are computed
    from the model's native normalized pressure output (pressure / |V_inf|^2
    is already the pressure-coefficient scaling) — this is correct as-is
    and must NOT be rescaled to physical units (that would make a
    "dimensionless" coefficient depend on each sample's flow speed).

  - Velocity/pressure fields are rescaled back to physical units (m/s, and
    kinematic pressure m^2/s^2) using each sample's v_inf_mag from
    stats.npz before any error is computed, since normalized-space errors
    aren't comparable across samples with different |V_inf|.

Primary metric is RELATIVE L2 ERROR per sample (||pred-targ||_2 /
||targ||_2), matching how the AirfRANS / NeurIPS 2024 ML4CFD competition
defines its per-quantity accuracy metric. This is scale-invariant, so it's
identical whether computed in normalized or physical units — but physical
units are still used for the RMS numbers and plots, since those are meant
to answer "how far off are we in real terms."

Scoring methodology follows the NeurIPS 2024 ML4CFD competition (built on
the AirfRANS/LIPS benchmark, Yagoubi et al., "NeurIPS 2024 ML4CFD
Competition: Results and Retrospective Analysis", arXiv:2506.08516):

  For each quantity, each sample's relative error e is classified against
  two thresholds (T1 < T2):
      e < T1        -> "great"        (2 points)
      T1 <= e < T2  -> "acceptable"   (1 point)
      e >= T2       -> "unacceptable" (0 points)
  Score_Accuracy = (2*Ng + 1*No + 0*Nr) / (2*N)

  The competition's full Global Score is
      Score = 0.4*Score_ML + 0.3*Score_OOD + 0.3*Score_Physics
      Score_ML = 0.75*Score_Accuracy + 0.25*Score_Speedup
  We can only compute the accuracy piece: Score_Speedup needs a timed
  reference-solver comparison we don't have, and Score_OOD needs a held-out
  *out-of-distribution* split (different AoA/Reynolds range set aside
  specifically for it) that this project's data pipeline doesn't produce.
  Those two are reported as "not computed" rather than guessed at — a
  fabricated number would be worse than none. What IS computed — the
  per-quantity accuracy classification and Score_Accuracy for both the flow
  fields and the Cl/Cd physics-compliance metrics — is the core "are these
  predictions good" signal and uses the paper's own thresholds and formula.

  Thresholds (Table 3 / Appendix B of arXiv:2506.08516 — the paper notes
  these are from the competition's preliminary edition and may have shifted
  slightly by the final edition, so treat this Score_Accuracy as a
  well-defined but approximate stand-in for the official leaderboard score,
  not a guaranteed match to it):
      u_x, u_y      : T1=0.10  T2=0.20
      p (volume)    : T1=0.02  T2=0.10
      p_s (surface) : T1=0.08  T2=0.20
      C_D           : T1=1.0   T2=10.0
      C_L           : T1=0.2   T2=0.5
      rho_D (Spearman rank corr, drag)  : T1=0.50  T2=0.80
      rho_L (Spearman rank corr, lift)  : T1=0.94  T2=0.98
  (nu_t, turbulent viscosity, is part of the official metric set but this
  model doesn't predict it, so it's excluded.)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import (
    AirfRANSGATrDataset, set_seed, SEED, VAL_FRACTION, CHECKPOINT_DIR, DATA_DIR,
    GRID_BOUNDS, GRID_RESOLUTION, GRID_STRETCH_CENTER, GRID_STRETCH_GAMMA,
)
from pipeline import FlowFieldPipeline
from prepare_gatr import SDF_LOG_EPS

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = CHECKPOINT_DIR / "best.pt"
BOUNDARY_SDF_THRESHOLD = 0.5  # points with |SDF| < this are on/near boundary
N_DETAIL_PLOTS = 5            # representative per-sample field plots (min/p25/median/p75/max error)
EPS = 1e-8

# (T1, T2): error < T1 -> great, T1 <= error < T2 -> acceptable, error >= T2 -> unacceptable
THRESHOLDS = {
    'u_x': (0.10, 0.20),
    'u_y': (0.10, 0.20),
    'p':   (0.02, 0.10),
    'p_s': (0.08, 0.20),
    'Cd':  (1.0, 10.0),
    'Cl':  (0.2, 0.5),
    'rho_D': (0.50, 0.80),
    'rho_L': (0.94, 0.98),
}


def load_model(checkpoint_path: Path) -> nn.Module:
    model = FlowFieldPipeline(
        input_scalar_dim=6,
        mv_channels=4,
        scalar_channels=8,
        n_heads=2,
        n_encoder_layers=4,
        grid_resolution=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        grid_stretch_center=GRID_STRETCH_CENTER,
        grid_stretch_gamma=GRID_STRETCH_GAMMA,
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


def relative_l2_error(pred: np.ndarray, targ: np.ndarray) -> float:
    """||pred - targ||_2 / ||targ||_2 — scale-invariant, matches the AirfRANS
    competition's per-quantity accuracy metric. Identical whether pred/targ
    are in normalized or physical units, since the scale cancels."""
    return float(np.linalg.norm(pred - targ) / (np.linalg.norm(targ) + EPS))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, computed without a scipy dependency:
    Pearson correlation of the ranks. Used for rho_D / rho_L — the
    competition uses rank correlation for Cd/Cl instead of relative error
    because Cd/Cl can be near zero, which makes relative error blow up."""
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    if np.std(rank_a) < EPS or np.std(rank_b) < EPS:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def classify(e: float, thresholds: tuple) -> str:
    t1, t2 = thresholds
    if e < t1:
        return 'great'
    elif e < t2:
        return 'acceptable'
    else:
        return 'unacceptable'


def compute_sample_metrics(
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
    """Per-sample metrics: relative L2 errors (scale-invariant, for scoring),
    Cl/Cd (from normalized pressure — dimensionless coefficients), and
    physical-unit RMS errors (for interpretability / plotting)."""

    boundary_mask = np.abs(sdf) < BOUNDARY_SDF_THRESHOLD
    n_b = normals[boundary_mask]
    p_pred_b_norm = pressure_pred_norm[boundary_mask]
    p_targ_b_norm = pressure_targ_norm[boundary_mask]

    # Force coefficients: unweighted average of Cp*n over boundary points
    # (boundary points were farthest-point-sampled -> roughly arc-length
    # uniform, so this approximates the true surface integral). Pressure-drag
    # only — omits the viscous/wall-shear contribution to Cd.
    Cl_pred = float(np.mean(p_pred_b_norm * n_b[:, 1])) if len(n_b) else 0.0
    Cl_targ = float(np.mean(p_targ_b_norm * n_b[:, 1])) if len(n_b) else 0.0
    Cd_pred = float(np.mean(p_pred_b_norm * n_b[:, 0])) if len(n_b) else 0.0
    Cd_targ = float(np.mean(p_targ_b_norm * n_b[:, 0])) if len(n_b) else 0.0

    p_pred_phys_b = pressure_pred_phys[boundary_mask]
    p_targ_phys_b = pressure_targ_phys[boundary_mask]

    return {
        'e_ux': relative_l2_error(velocity_pred_norm[:, 0], velocity_targ_norm[:, 0]),
        'e_uy': relative_l2_error(velocity_pred_norm[:, 1], velocity_targ_norm[:, 1]),
        'e_p': relative_l2_error(pressure_pred_norm, pressure_targ_norm),
        'e_ps': relative_l2_error(p_pred_b_norm, p_targ_b_norm) if len(n_b) else float('nan'),
        'Cl_pred': Cl_pred, 'Cl_targ': Cl_targ,
        'Cd_pred': Cd_pred, 'Cd_targ': Cd_targ,
        'v_rms': float(np.sqrt(np.mean((velocity_pred_phys - velocity_targ_phys) ** 2))),
        'p_rms': float(np.sqrt(np.mean((p_pred_phys_b - p_targ_phys_b) ** 2))) if len(n_b) else float('nan'),
        'n_boundary': int(np.sum(boundary_mask)),
    }


# SDF magnitude bands used to stratify field error by distance from the
# airfoil surface. RANS meshes are heavily refined right at the wall (to
# resolve the boundary layer), so a huge fraction of the full ~180k-point
# mesh sits in the first band — a single pooled relative-L2 error over the
# whole mesh is dominated by however well the model does there, and hides
# whether the rest of the field is actually fine.
SDF_BANDS = [
    (0.0, 0.01, "|sdf|<0.01 (near-wall)"),
    (0.01, 0.05, "0.01<=|sdf|<0.05"),
    (0.05, 0.1, "0.05<=|sdf|<0.1"),
    (0.1, 0.5, "0.1<=|sdf|<0.5"),
    (0.5, float('inf'), "|sdf|>=0.5 (far-field)"),
]


def new_band_accumulator() -> dict:
    return {
        label: {'sq_err_ux': 0.0, 'sq_targ_ux': 0.0,
                'sq_err_uy': 0.0, 'sq_targ_uy': 0.0,
                'sq_err_p': 0.0, 'sq_targ_p': 0.0,
                'n_points': 0}
        for _, _, label in SDF_BANDS
    }


def accumulate_band_stats(
    band_stats: dict,
    sdf: np.ndarray,
    velocity_pred_norm: np.ndarray,
    velocity_targ_norm: np.ndarray,
    pressure_pred_norm: np.ndarray,
    pressure_targ_norm: np.ndarray,
) -> None:
    """Pool squared-error / squared-target sums per SDF band across every
    point of every sample — relative L2 error is a global norm ratio, not a
    mean of per-sample ratios, so bands with few points in one sample still
    get folded in correctly at the end instead of being averaged unweighted."""
    abs_sdf = np.abs(sdf)
    for lo, hi, label in SDF_BANDS:
        mask = (abs_sdf >= lo) & (abs_sdf < hi)
        if not mask.any():
            continue
        b = band_stats[label]
        b['sq_err_ux'] += float(np.sum((velocity_pred_norm[mask, 0] - velocity_targ_norm[mask, 0]) ** 2))
        b['sq_targ_ux'] += float(np.sum(velocity_targ_norm[mask, 0] ** 2))
        b['sq_err_uy'] += float(np.sum((velocity_pred_norm[mask, 1] - velocity_targ_norm[mask, 1]) ** 2))
        b['sq_targ_uy'] += float(np.sum(velocity_targ_norm[mask, 1] ** 2))
        b['sq_err_p'] += float(np.sum((pressure_pred_norm[mask] - pressure_targ_norm[mask]) ** 2))
        b['sq_targ_p'] += float(np.sum(pressure_targ_norm[mask] ** 2))
        b['n_points'] += int(mask.sum())


def print_band_breakdown(band_stats: dict) -> None:
    """Relative L2 error per SDF band, pooled over every point/sample in
    that band — shows whether error is concentrated near the wall (mesh-
    refinement artifact) or spread through the field (a real model gap)."""
    total_points = sum(b['n_points'] for b in band_stats.values())
    print("\nRelative L2 error by distance-to-surface band "
          "(pooled over all points/samples in each band):")
    print(f"  {'band':26s}  {'n_points':>10s}  {'%':>6s}  {'e_ux':>8s}  {'e_uy':>8s}  {'e_p':>8s}")
    for _, _, label in SDF_BANDS:
        b = band_stats[label]
        n = b['n_points']
        if n == 0:
            print(f"  {label:26s}  {'(no points)':>10s}")
            continue
        e_ux = (b['sq_err_ux'] / (b['sq_targ_ux'] + EPS)) ** 0.5
        e_uy = (b['sq_err_uy'] / (b['sq_targ_uy'] + EPS)) ** 0.5
        e_p = (b['sq_err_p'] / (b['sq_targ_p'] + EPS)) ** 0.5
        pct = 100 * n / total_points if total_points else 0.0
        print(f"  {label:26s}  {n:10d}  {pct:5.1f}%  {e_ux:8.4f}  {e_uy:8.4f}  {e_p:8.4f}")


def print_distribution(name: str, values: np.ndarray, unit: str = "") -> None:
    """Report median/percentiles instead of mean±std for these (non-negative,
    right-skewed) error metrics — mean±std implies a symmetric interval that
    can extend below zero, which is meaningless for an error that can't be
    negative. The percentiles describe the actual, usually skewed, shape."""
    v = values[~np.isnan(values)]
    print(f"  {name}:")
    print(f"    median={np.median(v):.5f}{unit}  mean={np.mean(v):.5f}{unit}  std={np.std(v):.5f}{unit}")
    print(f"    [p5={np.percentile(v, 5):.5f}, p25={np.percentile(v, 25):.5f}, "
          f"p75={np.percentile(v, 75):.5f}, p95={np.percentile(v, 95):.5f}]  "
          f"min={np.min(v):.5f}  max={np.max(v):.5f}")


def plot_error_distributions(all_metrics: dict, pooled_v_err: np.ndarray, pooled_p_err: np.ndarray,
                              output_dir: Path) -> None:
    """Histograms of the error distribution across the whole val set — this
    is what mean±std can't show you: whether errors are tightly clustered,
    long-tailed, bimodal, or dominated by a few bad samples."""
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(f"Error Distributions — {len(all_metrics['e_p'])} held-out validation samples", fontsize=13)

    def hist(ax, data, title, xlabel, color='steelblue'):
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        ax.hist(data, bins=40, color=color, edgecolor='black', alpha=0.75)
        ax.axvline(np.median(data), color='red', linestyle='--', linewidth=1.5, label=f'median={np.median(data):.4f}')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("# samples")
        ax.legend(fontsize=8)

    hist(axes[0, 0], all_metrics['e_ux'], "Relative L2 error — v_x", "relative error")
    hist(axes[0, 1], all_metrics['e_uy'], "Relative L2 error — v_y", "relative error")
    hist(axes[0, 2], all_metrics['e_p'], "Relative L2 error — pressure (volume)", "relative error")

    hist(axes[1, 0], all_metrics['e_ps'], "Relative L2 error — pressure (surface)", "relative error")
    hist(axes[1, 1], [c['Cl_pred'] - c['Cl_targ'] for c in zip_metrics(all_metrics)], "Cl error (pred - target)",
         "Cl error", color='darkorange')
    hist(axes[1, 2], [c['Cd_pred'] - c['Cd_targ'] for c in zip_metrics(all_metrics)], "Cd error (pred - target)",
         "Cd error", color='darkorange')

    hist(axes[2, 0], all_metrics['v_rms'], "Velocity RMS error per sample", "RMS error (m/s)", color='seagreen')
    hist(axes[2, 1], all_metrics['p_rms'], "Pressure RMS error per sample", "RMS error (m²/s²)", color='seagreen')

    # Pooled point-wise errors (every point from every val sample, physical units) —
    # the actual per-point error distribution, not a per-sample summary.
    axes[2, 2].hist(pooled_v_err, bins=80, color='slategray', edgecolor='none', alpha=0.7, label='|v_pred|-|v_targ| (m/s)')
    axes[2, 2].axvline(0, color='black', linewidth=1)
    axes[2, 2].set_title("Pooled point-wise velocity-magnitude error\n(every point, every val sample)")
    axes[2, 2].set_xlabel("error (m/s)")
    axes[2, 2].set_ylabel("# points")
    axes[2, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "error_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved error distribution histograms to {output_dir / 'error_distributions.png'}")

    # Separate figure for the pooled pressure error (different units/scale than velocity)
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.hist(pooled_p_err, bins=80, color='indianred', edgecolor='none', alpha=0.75)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title("Pooled point-wise pressure error (every point, every val sample)")
    ax.set_xlabel("error (m²/s²)")
    ax.set_ylabel("# points")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution_pressure_pooled.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pooled pressure error histogram to {output_dir / 'error_distribution_pressure_pooled.png'}")


def zip_metrics(all_metrics: dict):
    """Reconstruct per-sample dicts from the column-oriented aggregator for
    convenience in plotting."""
    n = len(all_metrics['e_p'])
    keys = list(all_metrics.keys())
    return [{k: all_metrics[k][i] for k in keys} for i in range(n)]


def plot_sample_predictions(
    coords: np.ndarray,
    sdf: np.ndarray,
    velocity_pred_phys: np.ndarray,
    velocity_targ_phys: np.ndarray,
    pressure_pred_phys: np.ndarray,
    pressure_targ_phys: np.ndarray,
    sample_metrics: dict,
    sample_idx: int,
    v_inf_mag: float,
    role: str,
    output_dir: Path,
) -> None:
    """Visualize predicted vs target fields (physical units) for one sample."""
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample {sample_idx} [{role}] — |V_inf|={v_inf_mag:.2f} m/s\n"
                 f"e_p (relative L2, pressure)={sample_metrics['e_p']:.4f}  "
                 f"Cl_pred={sample_metrics['Cl_pred']:.4f} (target={sample_metrics['Cl_targ']:.4f})  "
                 f"Cd_pred={sample_metrics['Cd_pred']:.4f} (target={sample_metrics['Cd_targ']:.4f})",
                 fontsize=11)

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
    axes[0, 2].set_title(f"Velocity Error (RMS={sample_metrics['v_rms']:.4f} m/s)")
    axes[0, 2].set_aspect('equal')
    plt.colorbar(sc, ax=axes[0, 2], label='|Error| (m/s)')

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
    axes[1, 2].set_title(f"Pressure Error (RMS={sample_metrics['p_rms']:.4f} m²/s²)")
    axes[1, 2].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1, 2], label='|Error| (m²/s²)')

    plt.tight_layout()
    fname = output_dir / f"sample_{sample_idx:05d}_{role}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {fname}")


def compute_airfrans_score(all_metrics: dict, rho_D: float, rho_L: float) -> None:
    """Score_Accuracy per the NeurIPS 2024 ML4CFD competition formula, split
    into the field-accuracy group and the physics-compliance (Cd/Cl) group,
    since the official scoring keeps these as separate categories feeding
    Score_ML and Score_Physics respectively."""

    def bucket_counts(errors, thresholds):
        errors = np.asarray(errors)
        errors = errors[~np.isnan(errors)]
        labels = [classify(e, thresholds) for e in errors]
        return labels.count('great'), labels.count('acceptable'), labels.count('unacceptable')

    print("\n" + "=" * 80)
    print("AIRFRANS / NEURIPS 2024 ML4CFD ACCURACY SCORE")
    print("(source: Yagoubi et al., arXiv:2506.08516, Table 3 / Appendix B thresholds)")
    print("=" * 80)

    field_quantities = {'u_x': all_metrics['e_ux'], 'u_y': all_metrics['e_uy'],
                         'p': all_metrics['e_p'], 'p_s': all_metrics['e_ps']}
    Ng_field = No_field = Nr_field = 0
    for name, errs in field_quantities.items():
        g, o, r = bucket_counts(errs, THRESHOLDS[name])
        Ng_field += g; No_field += o; Nr_field += r
        n = g + o + r
        print(f"  {name:5s}: great={g:4d} ({g/n:.1%})  acceptable={o:4d} ({o/n:.1%})  "
              f"unacceptable={r:4d} ({r/n:.1%})")
    N_field = Ng_field + No_field + Nr_field
    score_accuracy_ml = (2 * Ng_field + No_field) / (2 * N_field)
    print(f"  -> Score_Accuracy (fields, feeds Score_ML) = {score_accuracy_ml:.4f}  "
          f"(Score_ML = 0.75*this + 0.25*Score_Speedup; Speedup not computed)")

    Cl_errs = [abs(a - b) / (abs(b) + EPS) for a, b in zip(all_metrics['Cl_pred'], all_metrics['Cl_targ'])]
    Cd_errs = [abs(a - b) / (abs(b) + EPS) for a, b in zip(all_metrics['Cd_pred'], all_metrics['Cd_targ'])]

    print()
    g, o, r = bucket_counts(Cl_errs, THRESHOLDS['Cl'])
    n = g + o + r
    print(f"  Cl   : great={g:4d} ({g/n:.1%})  acceptable={o:4d} ({o/n:.1%})  unacceptable={r:4d} ({r/n:.1%})")
    Ng_phys, No_phys, Nr_phys = g, o, r

    g, o, r = bucket_counts(Cd_errs, THRESHOLDS['Cd'])
    n = g + o + r
    print(f"  Cd   : great={g:4d} ({g/n:.1%})  acceptable={o:4d} ({o/n:.1%})  unacceptable={r:4d} ({r/n:.1%})")
    Ng_phys += g; No_phys += o; Nr_phys += r

    # rho thresholds are on the correlation value itself (higher=better), not an error —
    # classify manually: rho >= T2 -> great, T1 <= rho < T2 -> acceptable, rho < T1 -> unacceptable
    def classify_corr(rho, thresholds):
        t1, t2 = thresholds
        if rho >= t2:
            return 'great'
        elif rho >= t1:
            return 'acceptable'
        else:
            return 'unacceptable'

    rho_D_class = classify_corr(rho_D, THRESHOLDS['rho_D'])
    rho_L_class = classify_corr(rho_L, THRESHOLDS['rho_L'])
    print(f"  rho_D (Spearman rank corr, drag) = {rho_D:.4f}  -> {rho_D_class}")
    print(f"  rho_L (Spearman rank corr, lift) = {rho_L:.4f}  -> {rho_L_class}")
    for cls in (rho_D_class, rho_L_class):
        if cls == 'great':
            Ng_phys += 1
        elif cls == 'acceptable':
            No_phys += 1
        else:
            Nr_phys += 1

    N_phys = Ng_phys + No_phys + Nr_phys
    score_physics = (2 * Ng_phys + No_phys) / (2 * N_phys)
    print(f"  -> Score_Physics (Cd, Cl, rho_D, rho_L) = {score_physics:.4f}")

    print(f"\n  Score_OOD: NOT COMPUTED — requires a dedicated out-of-distribution split")
    print(f"  (different AoA/Reynolds range held out specifically for OOD testing),")
    print(f"  which prepare_gatr.py / train.py don't produce; this project only has an")
    print(f"  in-distribution random val split.")
    print(f"  Score_Speedup: NOT COMPUTED — requires timing a reference RANS solver run")
    print(f"  for comparison, which hasn't been benchmarked here.")
    print(f"  => Full Global Score (0.4*Score_ML + 0.3*Score_OOD + 0.3*Score_Physics) cannot")
    print(f"  be honestly computed without those two pieces. Score_Accuracy_fields=")
    print(f"  {score_accuracy_ml:.4f} and Score_Physics={score_physics:.4f} are the two")
    print(f"  components you do have.")
    print("=" * 80)


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}\n")

    dataset = AirfRANSGATrDataset(DATA_DIR)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    _, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Loaded {len(dataset)} total samples from {DATA_DIR} — "
          f"evaluating on ALL {len(val_set)} held-out validation samples\n")

    model = load_model(MODEL_PATH)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    all_metrics = {k: [] for k in ['e_ux', 'e_uy', 'e_p', 'e_ps', 'Cl_pred', 'Cl_targ',
                                    'Cd_pred', 'Cd_targ', 'v_rms', 'p_rms']}
    sample_records = []  # (global_idx, v_inf_mag, coords_np, sdf, phys fields, metrics) for plotting later
    pooled_v_err, pooled_p_err = [], []
    band_stats = new_band_accumulator()

    print(f"Evaluating {len(val_set)} samples...")
    with torch.no_grad():
        for loop_i, (coords, node_scalars, target) in enumerate(val_loader):
            global_idx = val_set.indices[loop_i]
            sample_dir = dataset.sample_dirs[global_idx]

            coords = coords.squeeze(0).to(DEVICE)
            node_scalars = node_scalars.squeeze(0).to(DEVICE)
            target = target.squeeze(0).to(DEVICE)

            fno_out = model.encode_to_grid(coords, node_scalars)

            query_coords = coords
            # node_scalars[:, 2] is sdf_log (see train.py) — invert
            # encode_wall_normal's asinh stretch back to raw sdf for the
            # physical sdf-banded diagnostics below.
            sdf_log_np = node_scalars[:, 2].cpu().numpy()
            sdf = np.sign(sdf_log_np) * np.sinh(np.abs(sdf_log_np)) * SDF_LOG_EPS
            normals = node_scalars[:, 3:5].cpu().numpy()
            target_np = target.cpu().numpy()

            pred = model.decode(fno_out, query_coords)

            coords_np = query_coords.cpu().numpy()
            pred_np = pred.cpu().numpy()

            velocity_pred_norm = pred_np[:, :2]
            velocity_targ_norm = target_np[:, :2]
            pressure_pred_norm = pred_np[:, 2]
            pressure_targ_norm = target_np[:, 2]

            stats = np.load(sample_dir / "stats.npz")
            v_inf_mag = float(stats['v_inf_mag'])

            velocity_pred_phys = velocity_pred_norm * v_inf_mag
            velocity_targ_phys = velocity_targ_norm * v_inf_mag
            pressure_pred_phys = pressure_pred_norm * (v_inf_mag ** 2)
            pressure_targ_phys = pressure_targ_norm * (v_inf_mag ** 2)

            m = compute_sample_metrics(
                coords_np, sdf, normals,
                velocity_pred_norm, velocity_targ_norm,
                pressure_pred_norm, pressure_targ_norm,
                velocity_pred_phys, velocity_targ_phys,
                pressure_pred_phys, pressure_targ_phys,
            )
            for k in all_metrics:
                all_metrics[k].append(m[k])

            accumulate_band_stats(
                band_stats, sdf,
                velocity_pred_norm, velocity_targ_norm,
                pressure_pred_norm, pressure_targ_norm,
            )

            v_mag_pred = np.linalg.norm(velocity_pred_phys, axis=1)
            v_mag_targ = np.linalg.norm(velocity_targ_phys, axis=1)
            pooled_v_err.append(v_mag_pred - v_mag_targ)
            boundary_mask = np.abs(sdf) < BOUNDARY_SDF_THRESHOLD
            pooled_p_err.append(pressure_pred_phys[boundary_mask] - pressure_targ_phys[boundary_mask])

            sample_records.append(dict(
                global_idx=global_idx, v_inf_mag=v_inf_mag, coords_np=coords_np, sdf=sdf,
                velocity_pred_phys=velocity_pred_phys, velocity_targ_phys=velocity_targ_phys,
                pressure_pred_phys=pressure_pred_phys, pressure_targ_phys=pressure_targ_phys,
                metrics=m,
            ))

            if (loop_i + 1) % 20 == 0 or (loop_i + 1) == len(val_set):
                print(f"  ...{loop_i + 1}/{len(val_set)}")

    pooled_v_err = np.concatenate(pooled_v_err)
    pooled_p_err = np.concatenate(pooled_p_err)

    # ---- Full-distribution statistics ----
    print("\n" + "=" * 80)
    print(f"FULL VALIDATION SET STATISTICS (n={len(val_set)} samples)")
    print("These are RMS/relative-L2 errors — non-negative and typically right-skewed.")
    print("mean +/- std is NOT a symmetric interval around the mean (it can imply")
    print("negative values that are impossible for these metrics) — read the")
    print("percentiles below, or eval_plots/error_distributions.png, for the actual shape.")
    print("=" * 80)
    print("\nRelative L2 error (scale-invariant, dimensionless fraction of ||target||):")
    print_distribution("v_x", np.array(all_metrics['e_ux']))
    print_distribution("v_y", np.array(all_metrics['e_uy']))
    print_distribution("pressure (volume)", np.array(all_metrics['e_p']))
    print_distribution("pressure (surface)", np.array(all_metrics['e_ps']))
    print("\nPhysical-unit RMS error per sample:")
    print_distribution("velocity", np.array(all_metrics['v_rms']), unit=" m/s")
    print_distribution("pressure (surface)", np.array(all_metrics['p_rms']), unit=" m²/s²")

    print_band_breakdown(band_stats)

    Cl_pred_arr = np.array(all_metrics['Cl_pred'])
    Cl_targ_arr = np.array(all_metrics['Cl_targ'])
    Cd_pred_arr = np.array(all_metrics['Cd_pred'])
    Cd_targ_arr = np.array(all_metrics['Cd_targ'])
    print("\nLift coefficient Cl (dimensionless, pressure-drag-only approximation):")
    print_distribution("|Cl_pred - Cl_targ|", np.abs(Cl_pred_arr - Cl_targ_arr))
    print("Drag coefficient Cd (dimensionless, pressure-drag-only approximation):")
    print_distribution("|Cd_pred - Cd_targ|", np.abs(Cd_pred_arr - Cd_targ_arr))

    rho_D = spearman_corr(Cd_pred_arr, Cd_targ_arr)
    rho_L = spearman_corr(Cl_pred_arr, Cl_targ_arr)

    # ---- Plots ----
    plots_dir = Path("eval_plots")
    plot_error_distributions(all_metrics, pooled_v_err, pooled_p_err, plots_dir)

    # Representative per-sample field plots: min / p25 / median / p75 / max by pressure relative error
    e_p_arr = np.array(all_metrics['e_p'])
    order = np.argsort(e_p_arr)
    n = len(order)
    pick_positions = {
        'best': order[0],
        'p25': order[max(0, n // 4)],
        'median': order[n // 2],
        'p75': order[min(n - 1, 3 * n // 4)],
        'worst': order[-1],
    }
    print(f"\nSaving {len(pick_positions)} representative per-sample field plots "
          f"(by pressure relative-L2 error rank)...")
    for role, idx in pick_positions.items():
        r = sample_records[idx]
        plot_sample_predictions(
            r['coords_np'], r['sdf'],
            r['velocity_pred_phys'], r['velocity_targ_phys'],
            r['pressure_pred_phys'], r['pressure_targ_phys'],
            r['metrics'], r['global_idx'], r['v_inf_mag'], role, plots_dir,
        )

    compute_airfrans_score(all_metrics, rho_D, rho_L)


if __name__ == "__main__":
    main()
