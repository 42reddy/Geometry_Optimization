"""
Trains FlowFieldPipeline (GATr -> GridProjector -> FNO -> GridDecoder) on
prepared AirfRANS samples from data/airfrans_gatr/.

Sample schema (per sample_XXXXX/data.npz, written by prepare_gatr.py):
    coords     (N, 2)  point coordinates, chord-normalized (chord = 1), raw
    sdf        (N,)    signed distance to airfoil surface, raw
    sdf_log    (N,)    sign(sdf) * asinh(|sdf| / eps) — wall-normal coordinate
                        fed to the network instead of raw sdf: raw sdf
                        compresses the whole boundary layer into a tiny
                        numeric range near 0, while this stretches it (see
                        encode_wall_normal in prepare_gatr.py).
    normals    (N, 2)  unit surface normals, raw
    freestream (2,)    inlet velocity, nondimensionalized: freestream / |V_inf|
    velocity   (N, 2)  target velocity, nondimensionalized: velocity / |V_inf|
    pressure   (N,)    target pressure, nondimensionalized: pressure / |V_inf|^2
    log_v_inf  ()      log(|V_inf|) — the flow-speed magnitude that gets
                        divided out of `freestream` above; kept as a separate
                        scalar feature since the nondimensional flow shape
                        still depends on it (through Reynolds number).
sample_XXXXX/stats.npz holds v_inf_mag, the freestream speed used for the
nondimensionalization above (needed only to convert predictions back to
physical units — training itself operates entirely in nondimensional space,
so samples with very different inlet speeds/Reynolds numbers contribute
comparably to the loss instead of the fastest-flow samples dominating it).

Training and validation run on the ~9k-point boundary-aware downsampled
cloud in data.npz (prepare_gatr.py doesn't save the full-resolution mesh).
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pipeline import FlowFieldPipeline

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
# derivatives above. Computed via central differences on GridDecoder's
# output rather than torch.autograd.grad, because create_graph=True through
# F.grid_sample requires a double backward that PyTorch doesn't implement
# for grid_sampler_2d. A plain finite difference only needs one ordinary
# backward pass, and is cheap here since it reuses the same FNO grid output
# (only the lightweight decode step is repeated, not GATr/FNO). Sized to
# ~1/3 of a grid cell (grid spacing is ~0.0625-0.069 for the default
# GRID_RESOLUTION/GRID_BOUNDS) so the stencil stays local to the bilinear
# interpolant instead of spanning multiple cells.
PHYSICS_EPS = 0.02

MV_CHANNELS = 8
SCALAR_CHANNELS = 12
N_HEADS = 4
N_ENCODER_LAYERS = 8
# Grid covers the observed coordinate range (x: [-2.16, 4.23], y: [-1.62, 1.62]
# from a 20-sample check) with margin for the rest of the dataset.
# Resolution bumped from (64, 128) now that the grid is stretched (below) —
# a uniform grid this size has cells ~10x wider than the near-wall
# boundary-layer band the full mesh concentrates points in, so no amount of
# resolution alone (without stretching) would have fixed that; this increase
# mainly buys sharper representation everywhere else now that near-wall
# density is handled by the stretch. Raise further if GPU memory allows —
# GridProjector's bipartite attention cost scales with H*W*bipartite_k.
GRID_RESOLUTION = (192, 384)   # (H, W)
GRID_BOUNDS = ((-3.0, 5.0), (-2.2, 2.2))   # ((x_min, x_max), (y_min, y_max))
# Grid density is concentrated here (mid-chord, on the chord line) and falls
# off towards the domain edges — see grid_stretch.py. GRID_STRETCH_GAMMA=0
# recovers the old uniform grid.
GRID_STRETCH_CENTER = (0.5, 0.0)
GRID_STRETCH_GAMMA = 3.0
KNN_K = 16
BIPARTITE_K = 8
FNO_HIDDEN_CHANNELS = 32
FNO_LAYERS = 4
FNO_MODES = 16

VAL_EVERY = 1
# ================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AirfRANSGATrDataset(Dataset):
    """Reads prepared per-sample NPZ files (data.npz, written by
    prepare_gatr.py) and assembles GATrEncoder's node_scalars."""

    def __init__(self, data_dir: Path):
        self.sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if not self.sample_dirs:
            raise RuntimeError(f"No samples found in {data_dir} — run prepare_gatr.py first.")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        sample_dir = self.sample_dirs[idx]
        data = np.load(sample_dir / "data.npz")
        freestream = torch.from_numpy(data['freestream']).float()
        log_v_inf = torch.tensor(float(data['log_v_inf']))

        coords = torch.from_numpy(data['coords']).float()
        sdf_log = torch.from_numpy(data['sdf_log']).float()
        normals = torch.from_numpy(data['normals']).float()
        velocity = torch.from_numpy(data['velocity']).float()
        pressure = torch.from_numpy(data['pressure']).float()

        n = coords.shape[0]
        freestream_bc = freestream.unsqueeze(0).expand(n, -1)   # broadcast to every node
        log_v_inf_bc = log_v_inf.expand(n).unsqueeze(-1)        # broadcast to every node
        node_scalars = torch.cat(
            [freestream_bc, sdf_log.unsqueeze(-1), normals, log_v_inf_bc], dim=-1
        )   # (N, 6)
        target = torch.cat([velocity, pressure.unsqueeze(-1)], dim=-1)                   # (N, 3)

        return coords, node_scalars, target


def check_grid_bounds(dataset: AirfRANSGATrDataset, bounds: tuple, n_check: int = 20) -> None:
    (x_min, x_max), (y_min, y_max) = bounds
    xs, ys = [], []
    for i in range(min(n_check, len(dataset))):
        coords, _, _ = dataset[i]
        xs.append(coords[:, 0])
        ys.append(coords[:, 1])
    xs, ys = torch.cat(xs), torch.cat(ys)
    obs_x, obs_y = (xs.min().item(), xs.max().item()), (ys.min().item(), ys.max().item())
    print(f"Observed coordinate range (from {n_check} samples): x={obs_x}, y={obs_y}")
    if obs_x[0] < x_min or obs_x[1] > x_max or obs_y[0] < y_min or obs_y[1] > y_max:
        print(f"WARNING: GRID_BOUNDS {bounds} do not cover the observed coordinate range — "
              f"out-of-range points will be clamped to the grid edge by GridDecoder.")


def continuity_residual_loss(model, fno_out: torch.Tensor, coords: torch.Tensor, eps: float) -> torch.Tensor:
    """Central-difference estimate of d(v_x)/dx + d(v_y)/dy (2D incompressible
    mass conservation) at each point, decoded from the already-computed FNO
    grid field. Real flows near the airfoil aren't purely inviscid/
    incompressible (this is a RANS solve), so this is a soft prior nudging
    the field towards physical coherence, not an exact constraint.
    """
    ex = coords.new_tensor([eps, 0.0])
    ey = coords.new_tensor([0.0, eps])

    pred_xp = model.decode(fno_out, coords + ex)
    pred_xm = model.decode(fno_out, coords - ex)
    pred_yp = model.decode(fno_out, coords + ey)
    pred_ym = model.decode(fno_out, coords - ey)

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

    for step, (coords, node_scalars, target) in enumerate(pbar):
        coords = coords.squeeze(0).to(device)
        node_scalars = node_scalars.squeeze(0).to(device)
        target = target.squeeze(0).to(device)

        with torch.set_grad_enabled(train):
            fno_out = model.encode_to_grid(coords, node_scalars)
            pred = model.decode(fno_out, coords)   # query at the same points used as input

            physics_loss = pred.new_zeros(())
            if physics_weight > 0:
                physics_loss = continuity_residual_loss(model, fno_out, coords, physics_eps)

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

    dataset = AirfRANSGATrDataset(DATA_DIR)
    print(f"Training on ~9k downsampled meshes per sample")
    check_grid_bounds(dataset, GRID_BOUNDS)

    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    print(f"Train samples: {n_train}  Val samples: {n_val}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = FlowFieldPipeline(
        input_scalar_dim=6,
        mv_channels=MV_CHANNELS,
        scalar_channels=SCALAR_CHANNELS,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        grid_resolution=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
        grid_stretch_center=GRID_STRETCH_CENTER,
        grid_stretch_gamma=GRID_STRETCH_GAMMA,
        knn_k=KNN_K,
        bipartite_k=BIPARTITE_K,
        fno_hidden_channels=FNO_HIDDEN_CHANNELS,
        fno_layers=FNO_LAYERS,
        fno_modes=FNO_MODES,
        n_outputs=3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Linear warmup (stabilizes early training, when GATr/FNO weights are still
    # near their random init and large steps can destabilize the FFT-based
    # spectral layers) followed by cosine decay to zero over the remaining epochs.
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
