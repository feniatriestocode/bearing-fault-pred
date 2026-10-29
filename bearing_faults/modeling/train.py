import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import typer
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bearing_faults.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from bearing_faults.data_loader import BearingTensorDataset
from bearing_faults.modeling.cnn import BaselineCNN
from bearing_faults.modeling.vit_baseline import BaselineViT
from bearing_faults.modeling.vit_cross_attention import DualBranchCrossAttentionViT

app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def build_model(name: str) -> nn.Module:
  if name == "cnn":
    return BaselineCNN(in_channels=2, num_classes=4)
  if name == "vit":
    return BaselineViT(in_channels=2, num_classes=4)
  if name == "cross_attention":
    return DualBranchCrossAttentionViT(num_classes=4)
  raise ValueError(f"Unknown model: {name}")


# ----------------------------------------------------------------------
# dB conversion (applied per fold, AFTER pooling; tensors on disk stay linear)
# ----------------------------------------------------------------------
def fit_ref(train_tensors: np.ndarray, q: float = 99.9) -> np.ndarray:
  """Per-channel reference power [2] (vib, cur) from the TRAIN fold only."""
  return np.percentile(train_tensors, q, axis=(0, 2, 3)).astype(np.float64)


def to_db(
    x: np.ndarray,
    ref: np.ndarray,
    floor_db: tuple[float, float] = (-60.0, -60.0),
    relative: tuple[bool, bool] = (False, False),
) -> np.ndarray:
  """Linear power [N,2,H,W] -> dB, clipped to [floor, 0], scaled to [0, 1].

  relative[c]=True subtracts, per sample, the 99th percentile of that
  channel's dB map (removes the absolute level, e.g. load-dependent
  amplitude of the current), so only the shape of the spectrum remains.
  """
  floor = np.asarray(floor_db, dtype=np.float32)[None, :, None, None]
  db = 10.0 * np.log10(np.maximum(x, 1e-12) / ref[None, :, None, None])
  db = db.astype(np.float32)
  for c, rel in enumerate(relative):
    if rel:
      db[:, c] -= np.percentile(db[:, c], 99, axis=(1, 2), keepdims=True)
  return ((np.clip(db, floor, 0.0) - floor) / (-floor)).astype(np.float32)


def log_tensor_stats(tag: str, t: np.ndarray) -> None:
  for c, ch in enumerate(["vib", "cur"]):
    v = t[:, c]
    logger.info(
        f"{tag} {ch}: min={v.min():.2f} max={v.max():.2f} mean={v.mean():.2f}"
        f" @floor={(v == 0).mean():.1%} @ceil={(v == 1).mean():.1%}"
    )


def get_cond(batch, cond_mode: str) -> torch.Tensor:
  cond = batch["condition"].to(DEVICE)
  return torch.zeros_like(cond) if cond_mode == "zero" else cond


# Mixup function
def run_epoch_mixup(
    model,
    loader,
    criterion,
    optimizer=None,
    mixup_prob: float = 0.5,
    cond_mode: str = "normal",
) -> tuple[float, float]:
  is_train = optimizer is not None
  model.train() if is_train else model.eval()
  total_loss, correct, total = 0.0, 0, 0
  with torch.set_grad_enabled(is_train):
    for batch in loader:
      x, y, cond = (
          batch["scalogram"].to(DEVICE),
          batch["label"].to(DEVICE),
          get_cond(batch, cond_mode),
      )
      do_mixup = is_train and (torch.rand(1).item() < mixup_prob)
      if do_mixup:
        lam = float(np.random.beta(0.2, 0.2))
        idx = torch.randperm(x.size(0), device=DEVICE)
        x_mixed = lam * x + (1.0 - lam) * x[idx]
        cond_mixed = lam * cond + (1.0 - lam) * cond[idx]
        logits = model(x_mixed, condition=cond_mixed)
        loss = lam * criterion(logits, y) + (1.0 - lam) * criterion(
            logits, y[idx]
        )
      else:
        logits = model(x, condition=cond)
        loss = criterion(logits, y)
      if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      total_loss += loss.item() * x.size(0)
      correct += (logits.argmax(dim=-1) == y).sum().item()
      total += x.size(0)
  return total_loss / total, correct / total


@torch.no_grad()
def channel_ablation(model, loader, cond_mode: str, seed: int) -> dict:
  model.eval()
  xs, ys, cs = [], [], []
  for b in loader:
    xs.append(b["scalogram"]); ys.append(b["label"]); cs.append(get_cond(b, cond_mode).cpu())
  x, y, c = torch.cat(xs), torch.cat(ys), torch.cat(cs)
  g = torch.Generator().manual_seed(seed)

  idx = torch.empty(len(y), dtype=torch.long)
  for k in y.unique():
    src = (y != k).nonzero().squeeze(1)
    dst = (y == k).nonzero().squeeze(1)
    idx[dst] = src[torch.randint(len(src), (len(dst),), generator=g)]

  def acc(xin):
    correct = 0
    for i in range(0, len(y), 256):
      out = model(xin[i:i + 256].to(DEVICE), condition=c[i:i + 256].to(DEVICE))
      correct += (out.argmax(-1).cpu() == y[i:i + 256]).sum().item()
    return correct / len(y)

  x_cur, x_vib = x.clone(), x.clone()
  x_cur[:, 1] = x[idx, 1]   # ρεύμα από άλλη κλάση
  x_vib[:, 0] = x[idx, 0]   # vibration από άλλη κλάση
  return {"full": acc(x), "cur_other_class": acc(x_cur), "vib_other_class": acc(x_vib)}




def group_aware_split(df, class_col, group_col="source_file", seed=42, **_):
  rng = np.random.default_rng(seed)
  val_files = []
  for cls, g in df.groupby(class_col):
    files = sorted(g[group_col].unique())
    assert len(files) >= 2, f"class {cls}: need >=2 files in trainval, got {len(files)}"
    val_files.append(rng.choice(files))
  is_val = df[group_col].isin(val_files).values
  return np.where(~is_val)[0], np.where(is_val)[0]

def train_one_regime(
    model_name: str,
    df_ss: pd.DataFrame,
    tensors_ss: np.ndarray,
    test_regime: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    class_col: str,
    cond_mode: str,
    current_relative: bool,
    floor_vib: float,
    floor_cur: float,
    tag: str,
) -> dict:
  set_seed(seed)
  run_name = f"ABG_{model_name}_{tag}"

  is_test_regime = df_ss["regime_id"] == test_regime
  trainval_df, test_df = (
      df_ss[~is_test_regime].reset_index(drop=True),
      df_ss[is_test_regime].reset_index(drop=True),
  )
  trainval_tensors, test_tensors = (
      tensors_ss[~is_test_regime.values],
      tensors_ss[is_test_regime.values],
  )

  train_idx, val_idx = group_aware_split(trainval_df, class_col, seed=seed)
  train_df, val_df = trainval_df.iloc[train_idx].reset_index(
      drop=True
  ), trainval_df.iloc[val_idx].reset_index(drop=True)
  train_tensors, val_tensors = (
      trainval_tensors[train_idx],
      trainval_tensors[val_idx],
  )
  logger.info(
      f"[regime {test_regime}] val files={val_df['source_file'].nunique()},"
      f" val class counts={val_df[class_col].value_counts().to_dict()}"
  )

  # ---- dB with ref fitted on the TRAIN fold only ----
  ref = fit_ref(train_tensors)
  db_kwargs = dict(
      ref=ref,
      floor_db=(floor_vib, floor_cur),
      relative=(False, current_relative),
  )
  train_tensors = to_db(train_tensors, **db_kwargs)
  val_tensors = to_db(val_tensors, **db_kwargs)
  test_tensors = to_db(test_tensors, **db_kwargs)
  logger.info(
      f"[regime {test_regime}] ref vib={ref[0]:.3e}, cur={ref[1]:.3e},"
      f" floors=({floor_vib}, {floor_cur}), current_relative={current_relative},"
      f" cond_mode={cond_mode}"
  )
  for name, t in [
      ("train", train_tensors),
      ("val", val_tensors),
      ("test", test_tensors),
  ]:
    log_tensor_stats(f"[regime {test_regime}] {name}", t)
  # Where are the floor pixels of the current? (4 frequency bands, low->high)
  band_floor = (train_tensors[:, 1] == 0).mean(axis=(0, 2)).reshape(4, 16).mean(1)
  logger.info(
      f"[regime {test_regime}] current @floor per freq band (low->high):"
      f" {np.round(band_floor, 3).tolist()}"
  )

  train_loader = DataLoader(
      BearingTensorDataset(train_tensors, train_df, augment=True),
      batch_size=batch_size,
      shuffle=True,
  )
  val_loader = DataLoader(
      BearingTensorDataset(val_tensors, val_df, augment=False),
      batch_size=batch_size,
      shuffle=False,
  )
  test_loader = DataLoader(
      BearingTensorDataset(test_tensors, test_df, augment=False),
      batch_size=batch_size,
      shuffle=False,
  )
  
  net = build_model(model_name).to(DEVICE)

  # ΣΥΝΔΥΑΣΜΟΣ ΑΒΓ: Weight Decay 0.05 + Warmup & Cosine + Mixup
  optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
  warmup_epochs = 5
  warmup_sched = LinearLR(
      optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
  )
  cosine_sched = CosineAnnealingLR(
      optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
  )
  scheduler = SequentialLR(
      optimizer,
      schedulers=[warmup_sched, cosine_sched],
      milestones=[warmup_epochs],
  )

  criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
  writer = SummaryWriter(
      log_dir=str(
          PROJ_ROOT / "reports" / "runs" / f"{run_name}_testregime{test_regime}"
      )
  )
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  best_ckpt_path = MODELS_DIR / f"best_{run_name}_regime{test_regime}.pt"
  # the ref is part of the model: inference must use the same constants
  np.save(MODELS_DIR / f"ref_{run_name}_regime{test_regime}.npy", ref)

  best_val_acc = 0.0
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = run_epoch_mixup(
        net, train_loader, criterion, optimizer, mixup_prob=0.5,
        cond_mode=cond_mode,
    )
    val_loss, val_acc = run_epoch_mixup(
        net, val_loader, criterion, mixup_prob=0.0, cond_mode=cond_mode
    )
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    logger.info(
        f"[regime {test_regime}] Epoch {epoch:03d} | lr={current_lr:.2e} |"
        f" train_loss={train_loss:.4f} train_acc={train_acc:.4f} |"
        f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )
    writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(net.state_dict(), best_ckpt_path)

  net.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
  test_loss, test_acc = run_epoch_mixup(
      net, test_loader, criterion, mixup_prob=0.0, cond_mode=cond_mode
  )
  ablation = channel_ablation(net, test_loader, cond_mode, seed)

  from sklearn.metrics import confusion_matrix
  net.eval(); preds, labels = [], []
  with torch.no_grad():
    for b in test_loader:
        out = net(b["scalogram"].to(DEVICE), condition=get_cond(b, cond_mode))
        preds += out.argmax(-1).cpu().tolist(); labels += b["label"].tolist()
  logger.info(f"[regime {test_regime}] confusion matrix (rows=true):\n{confusion_matrix(labels, preds)}")
  cm = confusion_matrix(labels, preds)
  cm_dir = PROJ_ROOT / "reports" / "results" / "confusion"
  cm_dir.mkdir(parents=True, exist_ok=True)
  np.save(cm_dir / f"cm_{run_name}_regime{test_regime}.npy", cm)
  logger.success(
      f"[regime {test_regime}, unseen] test_loss={test_loss:.4f}"
      f" test_acc={test_acc:.4f} | ablation={ablation}"
  )
  writer.close()
  return {
      "regime": test_regime,
      "best_val_acc": best_val_acc,
      "test_acc": test_acc,
      **{f"abl_{k}": v for k, v in ablation.items()},
  }


@app.command()
def main(
    model: str = typer.Option(..., help="cnn | vit | cross_attention"),
    test_regime: int = typer.Option(None),
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 5e-5,
    seed: int = 42,
    class_col: str = typer.Option(
        "fault_class", help="CSV column with the class label (for stratifying)"
    ),
    cond_mode: str = typer.Option(
        "normal", help="normal | zero (zero = ablate the condition input)"
    ),
    current_relative: bool = typer.Option(
        False, help="Per-sample relative dB for the current channel"
    ),
    floor_vib: float = -60.0,
    floor_cur: float = -60.0,
    tag: str = typer.Option("db", help="Run label used in file/log names"),
    features_path: Path = PROCESSED_DATA_DIR / "full_dataset.csv",
    tensors_path: Path = PROCESSED_DATA_DIR / "bearing_tensors.npy",
):
  if cond_mode not in ("normal", "zero"):
    raise typer.BadParameter("cond_mode must be 'normal' or 'zero'")
  df = pd.read_csv(features_path)
  if class_col not in df.columns:
    raise typer.BadParameter(
        f"class_col '{class_col}' not in CSV columns: {list(df.columns)}"
    )
  tensors = np.load(tensors_path)
  assert len(df) == len(tensors), (
      f"CSV rows ({len(df)}) != tensors ({len(tensors)}): re-run dataset.py"
      " and features.py with the same window/stride"
  )
  ss_mask = (df["state"] == "Steady-State").values
  df_ss, tensors_ss = df[ss_mask].reset_index(drop=True), tensors[ss_mask]

  regimes_to_run = (
      [test_regime]
      if test_regime is not None
      else sorted(df_ss["regime_id"].unique().tolist())
  )
  results = [
      train_one_regime(
          model, df_ss, tensors_ss, regime, epochs, batch_size, lr, seed,
          class_col, cond_mode, current_relative, floor_vib, floor_cur, tag,
      )
      for regime in regimes_to_run
  ]

  res_df = pd.DataFrame(results).set_index("regime")
  out_dir = PROJ_ROOT / "reports" / "results"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"results_{model}_{tag}.csv"
  res_df.to_csv(out_path)
  logger.info(f"\n{res_df.round(4).to_string()}")
  logger.info(f"Saved per-regime results to {out_path}")

  test_accs = res_df["test_acc"].values
  logger.success(
      f"\n=== ABG {model} [{tag}]: Mean ± std: {test_accs.mean():.4f} ±"
      f" {test_accs.std():.4f} ==="
  )


if __name__ == "__main__":
    app()
