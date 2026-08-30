"""
Trains FlowFieldPipeline (GATr -> GridProjector -> FNO -> GridDecoder) on
prepared AirfRANS samples from data/airfrans_gatr/.

Sample schema (per sample_XXXXX/data.npz, written by prepare_gatr.py):
    coords     (N, 2)  point coordinates, chord-normalized (chord = 1), raw
    sdf        (N,)    signed distance to airfoil surface, raw
    normals    (N, 2)  unit surface normals, raw
    freestream (2,)    inlet velocity, nondimensionalized: freestream / |V_inf|
    velocity   (N, 2)  target velocity, nondimensionalized: velocity / |V_inf|
    pressure   (N,)    target pressure, nondimensionalized: pressure / |V_inf|^2
sample_XXXXX/stats.npz holds v_inf_mag, the freestream speed used for the
nondimensionalization above (needed only to convert predictions back to
physical units — training itself operates entirely in nondimensional space,
so samples with very different inlet speeds/Reynolds numbers contribute
comparably to the loss instead of the fastest-flow samples dominating it).

Training and validation loss are computed at the same points fed into the
encoder (the downsampled point cloud) — GridDecoder's continuous
interpolation means the trained model can equally be queried at any other
coordinates (e.g. the full original mesh) at evaluation time, but that is
not exercised here.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pipeline import FlowFieldPipeline

# CONFIG
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

MV_CHANNELS = 4
SCALAR_CHANNELS = 8
N_HEADS = 2
N_ENCODER_LAYERS = 4

GRID_RESOLUTION = (64, 128)   # (H, W)
GRID_BOUNDS = ((-3.0, 5.0), (-2.2, 2.2))   # ((x_min, x_max), (y_min, y_max))
KNN_K = 16
BIPARTITE_K = 8
FNO_HIDDEN_CHANNELS = 32
FNO_LAYERS = 4
FNO_MODES = 16

VAL_EVERY = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AirfRANSGATrDataset(Dataset):
    """Reads prepared per-sample NPZ files and assembles GATrEncoder's node_scalars."""

    def __init__(self, data_dir: Path):
        self.sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
        if not self.sample_dirs:
            raise RuntimeError(f"No samples found in {data_dir} — run prepare_gatr.py first.")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int):
        data = np.load(self.sample_dirs[idx] / "data.npz")
        coords = torch.from_numpy(data['coords']).float()
        sdf = torch.from_numpy(data['sdf']).float()
        normals = torch.from_numpy(data['normals']).float()
        freestream = torch.from_numpy(data['freestream']).float()
        velocity = torch.from_numpy(data['velocity']).float()
        pressure = torch.from_numpy(data['pressure']).float()

        n = coords.shape[0]
        freestream_bc = freestream.unsqueeze(0).expand(n, -1)   # broadcast to every node
        node_scalars = torch.cat([freestream_bc, sdf.unsqueeze(-1), normals], dim=-1)   # (N, 5)
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


def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    per_channel = ((pred - target) ** 2).mean(dim=0)   # (3,): v_x, v_y, pressure
    total = per_channel.mean()
    return total, per_channel.detach()


def run_epoch(model, loader, optimizer, device, epoch: int, train: bool):
    model.train(train)
    total_loss = 0.0
    total_per_channel = torch.zeros(3)
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
            pred = model(coords, node_scalars, coords)   # query at the same points used as input
            loss, per_channel = compute_loss(pred, target)

        if train:
            (loss / GRAD_ACCUM_STEPS).backward()
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        total_per_channel += per_channel.cpu()
        n_samples += 1

        pbar.set_postfix(loss=f"{total_loss / n_samples:.5f}")

    if train and n_samples % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / n_samples, total_per_channel / n_samples


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = AirfRANSGATrDataset(DATA_DIR)
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
        input_scalar_dim=5,
        mv_channels=MV_CHANNELS,
        scalar_channels=SCALAR_CHANNELS,
        n_heads=N_HEADS,
        n_encoder_layers=N_ENCODER_LAYERS,
        grid_resolution=GRID_RESOLUTION,
        grid_bounds=GRID_BOUNDS,
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

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_per_channel = run_epoch(model, train_loader, optimizer, device, epoch, train=True)
        scheduler.step()

        log = (f"Epoch {epoch:04d}  train_loss={train_loss:.5f} "
               f"(v_x={train_per_channel[0]:.5f} v_y={train_per_channel[1]:.5f} p={train_per_channel[2]:.5f})  "
               f"lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % VAL_EVERY == 0:
            val_loss, val_per_channel = run_epoch(model, val_loader, optimizer, device, epoch, train=False)
            log += (f"  val_loss={val_loss:.5f} "
                    f"(v_x={val_per_channel[0]:.5f} v_y={val_per_channel[1]:.5f} p={val_per_channel[2]:.5f})")

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
