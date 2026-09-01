"""
Trains GINOTModel (GeometryEncoder -> SolutionDecoder) on prepared AirfRANS
samples from data/airfrans_gatr/.

Sample schema (per sample_XXXXX/data.npz, written by prepare_gatr.py):
    coords         (N, 2)  query-point coordinates, chord-normalized (chord = 1)
    sdf            (N,)    signed distance to airfoil surface — kept for
                            eval.py's physical diagnostics only; not a model
                            input (GINOT captures geometry purely from
                            surface_coords, no SDF needed)
    normals        (N, 2)  unit surface normals — also diagnostics-only
    freestream     (2,)    inlet velocity, nondimensionalized: freestream / |V_inf|
    velocity       (N, 2)  target velocity, nondimensionalized: velocity / |V_inf|
    pressure       (N,)    target pressure, nondimensionalized: pressure / |V_inf|^2
    log_v_inf      ()      log(|V_inf|) — the flow-speed magnitude that gets
                           divided out of `freestream` above; kept as a separate
                           scalar feature since the nondimensional flow shape
                           still depends on it (through Reynolds number).
    surface_coords (S, 2)  full-resolution airfoil contour (sdf == 0), the
                           geometry encoder's input.

sample_XXXXX/stats.npz holds v_inf_mag, the freestream speed used for the
nondimensionalization above (needed only to convert predictions back to
physical units — training itself operates entirely in nondimensional space,
so samples with very different inlet speeds/Reynolds numbers contribute
comparably to the loss instead of the fastest-flow samples dominating it).

Training and validation query the ~9k-point boundary-aware downsampled cloud
in data.npz (prepare_gatr.py doesn't save the full-resolution volume mesh).
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pipeline import GINOTModel

# ==== CONFIG ====
DATA_DIR = Path("/kaggle/input/datasets/reddy42/gatr-aerfrans/airfrans_gatr")
CHECKPOINT_DIR = Path("checkpoints")
SEED = 0

VAL_FRACTION = 0.1
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_ACCUM_STEPS = 4          # samples per optimizer step (point-cloud graphs vary in size -> batch_size=1 loader)
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 5             # linear LR warmup before cosine decay — stabilizes early training
EARLY_STOP_PATIENCE = 20      # stop if val_loss hasn't improved in this many epochs

# Per-channel loss weights [v_x, v_y, pressure]. Equal weighting let the
# model minimize loss mostly by nailing v_x (largest-magnitude, smoothest
# channel) while neglecting v_y and pressure, which are harder but not
# proportionally louder in an unweighted MSE. Weighted up here so the
# optimizer can't ignore them.
LOSS_WEIGHTS = (1.0, 2.0, 3.0)

# Weight on the incompressible-continuity residual (d(v_x)/dx + d(v_y)/dy = 0
# at each query point). This is a soft physical constraint, not a replacement
# for the data loss — kept small so it regularizes the field towards a
# coherent (divergence-free) flow instead of dominating the fit to data.
# Set to 0.0 to disable.
PHYSICS_WEIGHT = 0.02
# Finite-difference step (chord units) used to estimate the spatial
# derivatives above. Computed via central differences on the decoder's
# output rather than torch.autograd.grad, to avoid a double-backward
# through attention; cheap since it reuses the same encoder context (only
# the lightweight decode step is repeated, not the geometry encoder).
PHYSICS_EPS = 0.02

# GINOT Architecture Config — Scaled Standard Configuration (Large)
EMBED_DIM = 384
N_HEADS = 6                   # Yields d_k = 64 per head (GPU-aligned)
FFN_MULT = 4                  # d_ffn = 1536
N_SELF_ATTN_LAYERS = 8        # Encoder self-attention depth
N_DECODER_LAYERS = 4          # Decoder cross-attention depth

# Spatial & Positional Encoding
N_CENTROIDS = 512             # Increased spatial resolution over airfoil chord
GROUP_RADIUS = 0.04           # Adjusted tighter for denser centroid sampling
GROUP_MAX_NEIGHBORS = 64      # Neighbor support size
PE_FREQS = 12                 # Frequency bands (yields 2 * 12 * 2 = 48 positional dims for 2D coords)

# Conditioning Sub-network
CONDITION_HIDDEN = 192        # Set to EMBED_DIM // 2 for balanced parameter modulation

VAL_EVERY = 1
# ================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AirfRANSDataset(Dataset):
    """Reads prepared per-sample NPZ files (data.npz, written by
    prepare_gatr.py): the airfoil surface cloud (geometry encoder input),
    the per-sample scalar condition, and the query points/targets."""

    def __init__(self, data_dir: Path):
        self.sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if not self.sample_dirs:
            raise RuntimeError(f"No samples found in {data_dir} — run prepare_gatr.py first.")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        data = np.load(sample_dir / "data.npz")

        surface_coords = torch.from_numpy(data['surface_coords']).float()
        freestream = torch.from_numpy(data['freestream']).float()
        log_v_inf = torch.tensor(float(data['log_v_inf']))
        condition = torch.cat([freestream, log_v_inf.unsqueeze(0)])   # (3,)

        coords = torch.from_numpy(data['coords']).float()
        velocity = torch.from_numpy(data['velocity']).float()
        pressure = torch.from_numpy(data['pressure']).float()
        target = torch.cat([velocity, pressure.unsqueeze(-1)], dim=-1)   # (N, 3)

        return surface_coords, coords, condition, target


def continuity_residual_loss(model, context: torch.Tensor, coords: torch.Tensor, eps: float) -> torch.Tensor:
    """Central-difference estimate of d(v_x)/dx + d(v_y)/dy (2D incompressible
    mass conservation) at each point, decoded from the already-computed
    encoder context. Real flows near the airfoil aren't purely inviscid/
    incompressible (this is a RANS solve), so this is a soft prior nudging
    the field towards physical coherence, not an exact constraint.
    """
    ex = coords.new_tensor([eps, 0.0])
    ey = coords.new_tensor([0.0, eps])

    pred_xp = model.decode(context, coords + ex)
    pred_xm = model.decode(context, coords - ex)
    pred_yp = model.decode(context, coords + ey)
    pred_ym = model.decode(context, coords - ey)

    dvx_dx = (pred_xp[:, 0] - pred_xm[:, 0]) / (2 * eps)
    dvy_dy = (pred_yp[:, 1] - pred_ym[:, 1]) / (2 * eps)
    continuity_residual = dvx_dx + dvy_dy
    return (continuity_residual ** 2).mean()


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    physics_loss: torch.Tensor,
    physics_weight: float,
) -> tuple:
    """Weighted per-channel MSE plus the (already-computed) physics penalty."""
    per_channel = ((pred - target) ** 2).mean(dim=0)   # (3,): v_x, v_y, pressure
    total = (per_channel * weights).sum() / weights.sum()
    total = total + physics_weight * physics_loss
    return total, per_channel.detach()


def run_epoch(model, loader, optimizer, device, epoch: int, train: bool,
              loss_weights: torch.Tensor, physics_weight: float, physics_eps: float):
    model.train(train)
    total_loss = 0.0
    total_per_channel = torch.zeros(3)
    total_physics_loss = 0.0
    n_samples = 0

    if train:
        optimizer.zero_grad()

    phase = "train" if train else "val"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{phase}]", leave=False, unit="sample")

    for step, (surface_coords, coords, condition, target) in enumerate(pbar):
        surface_coords = surface_coords.squeeze(0).to(device)
        coords = coords.squeeze(0).to(device)
        condition = condition.squeeze(0).to(device)
        target = target.squeeze(0).to(device)

        with torch.set_grad_enabled(train):
            context = model.encode(surface_coords, condition)
            pred = model.decode(context, coords)   # query at the same points used as input

            physics_loss = pred.new_zeros(())
            if physics_weight > 0:
                physics_loss = continuity_residual_loss(model, context, coords, physics_eps)

            loss, per_channel = compute_loss(pred, target, loss_weights, physics_loss, physics_weight)

        if train:
            (loss / GRAD_ACCUM_STEPS).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        total_per_channel += per_channel.cpu()
        total_physics_loss += physics_loss.item()
        n_samples += 1

        pbar.set_postfix(loss=f"{total_loss / n_samples:.5f}")

    if train and n_samples % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / n_samples, total_per_channel / n_samples, total_physics_loss / n_samples


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AirfRANSDataset(DATA_DIR)
    print(f"Training on ~9k downsampled query points per sample")

    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train samples: {n_train}  Val samples: {n_val}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = GINOTModel(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_self_attn_layers=N_SELF_ATTN_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_centroids=N_CENTROIDS,
        group_radius=GROUP_RADIUS,
        group_max_neighbors=GROUP_MAX_NEIGHBORS,
        pe_freqs=PE_FREQS,
        condition_dim=3,
        condition_hidden=CONDITION_HIDDEN,
        ffn_mult=FFN_MULT,
        n_outputs=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Linear warmup (stabilizes early training, when attention weights are
    # still near their random init and large steps can destabilize training)
    # followed by cosine decay to zero over the remaining epochs.
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    loss_weights = torch.tensor(LOSS_WEIGHTS, device=device)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_per_channel, train_physics = run_epoch(
            model, train_loader, optimizer, device, epoch, train=True,
            loss_weights=loss_weights, physics_weight=PHYSICS_WEIGHT, physics_eps=PHYSICS_EPS,
        )
        scheduler.step()

        log = (f"Epoch {epoch:04d}  train_loss={train_loss:.5f} "
               f"(v_x={train_per_channel[0]:.5f} v_y={train_per_channel[1]:.5f} p={train_per_channel[2]:.5f} "
               f"physics={train_physics:.5f})  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % VAL_EVERY == 0:
            val_loss, val_per_channel, val_physics = run_epoch(
                model, val_loader, optimizer, device, epoch, train=False,
                loss_weights=loss_weights, physics_weight=PHYSICS_WEIGHT, physics_eps=PHYSICS_EPS,
            )
            log += (f"  val_loss={val_loss:.5f} "
                    f"(v_x={val_per_channel[0]:.5f} v_y={val_per_channel[1]:.5f} p={val_per_channel[2]:.5f} "
                    f"physics={val_physics:.5f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
                log += "  [saved best]"
            else:
                epochs_since_improvement += 1

        print(log)

        if epochs_since_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nNo val improvement in {EARLY_STOP_PATIENCE} epochs — stopping early.")
            break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "last.pt")
    print(f"\nDone. Best val_loss={best_val_loss:.5f}. Checkpoints saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
