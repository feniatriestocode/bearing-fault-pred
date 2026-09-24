from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import typer
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bearing_faults.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from bearing_faults.modeling.cnn import BaselineCNN
from bearing_faults.data_loader import BearingTensorDataset
from bearing_faults.modeling.vit_baseline import BaselineViT
from bearing_faults.modeling.vit_cross_attention import DualBranchCrossAttentionViT

app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize01(x: torch.Tensor) -> torch.Tensor:
  """Per-image min-max σε [0,1], μόνο για TensorBoard προβολή."""
  x = x.clone()
  b = x.shape[0]
  flat = x.view(b, -1)
  mins = flat.min(dim=1, keepdim=True).values
  maxs = flat.max(dim=1, keepdim=True).values
  flat = (flat - mins) / (maxs - mins + 1e-8)
  return flat.view_as(x)


def build_model(name: str) -> nn.Module:
  if name == "cnn":
    return BaselineCNN(in_channels=2, num_classes=4)
  if name == "vit":
    return BaselineViT(in_channels=2, num_classes=4)
  if name == "cross_attention":
    return DualBranchCrossAttentionViT(num_classes=4)
  raise ValueError(f"Unknown model: {name}")


def run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, float]:
  is_train = optimizer is not None
  model.train() if is_train else model.eval()

  total_loss, correct, total = 0.0, 0, 0
  with torch.set_grad_enabled(is_train):
    for batch in loader:
      x = batch["scalogram"].to(DEVICE)
      y = batch["label"].to(DEVICE)
      cond = batch["condition"].to(DEVICE)

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


def group_aware_split(df: pd.DataFrame, group_col: str = "source_file", test_size: float = 0.2, seed: int = 42):
  """Χωρίζει σε train/val έτσι ώστε όλα τα windows του ΙΔΙΟΥ αρχείου
  (group_col) να πάνε ΟΛΑ μαζί στο ίδιο set - ποτέ σκορπισμένα. Απαραίτητο
  με overlapping windows (stride < window_size), όπου γειτονικά windows
  είναι σχεδόν πανομοιότυπα· τυχαίο split ανά window θα έδινε data leakage
  (τεχνητά διογκωμένο validation accuracy).
  """
  splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
  train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
  return train_idx, val_idx


def train_one_regime(
    model_name: str,
    df_ss: pd.DataFrame,
    tensors_ss: np.ndarray,
    test_regime: int,
    epochs: int,
    batch_size: int,
    lr: float,
    log_images: bool,
) -> dict:
  """Εκπαιδεύει από την αρχή, με το `test_regime` εντελώς έξω. Επιστρέφει
  metrics dict (best_val_acc, test_acc) για αυτόν τον γύρο.
  """
  is_test_regime = df_ss["regime_id"] == test_regime
  trainval_df = df_ss[~is_test_regime].reset_index(drop=True)
  trainval_tensors = tensors_ss[~is_test_regime.values]
  test_df = df_ss[is_test_regime].reset_index(drop=True)
  test_tensors = tensors_ss[is_test_regime.values]

  train_idx, val_idx = group_aware_split(trainval_df)
  train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
  train_tensors = trainval_tensors[train_idx]
  val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
  val_tensors = trainval_tensors[val_idx]

  logger.info(
      f"[regime {test_regime} έξω] Train: {len(train_df)} (από "
      f"{train_df['source_file'].nunique()} αρχεία) | Val: {len(val_df)} "
      f"(από {val_df['source_file'].nunique()} αρχεία) | "
      f"Test: {len(test_df)}"
  )

  train_ds = BearingTensorDataset(train_tensors, train_df, augment=True)
  val_ds = BearingTensorDataset(val_tensors, val_df, augment=False)
  test_ds = BearingTensorDataset(test_tensors, test_df, augment=False)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

  net = build_model(model_name).to(DEVICE)
  optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.5)

  run_name = f"{model_name}_testregime{test_regime}"
  writer = SummaryWriter(log_dir=str(PROJ_ROOT / "reports" / "runs" / run_name))

  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  best_ckpt_path = MODELS_DIR / f"best_{model_name}_regime{test_regime}.pt"

  best_val_acc = 0.0
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = run_epoch(net, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(net, val_loader, criterion)

    logger.info(
        f"[regime {test_regime}] Epoch {epoch:03d} | train_loss={train_loss:.4f} "
        f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )
    writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
    writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

    if epoch == 1 and log_images:
      sample_x = next(iter(train_loader))["scalogram"][:8]
      writer.add_images("Sample/vibration_scalograms", _normalize01(sample_x[:, 0:1]), epoch)
      writer.add_images("Sample/current_scalograms", _normalize01(sample_x[:, 1:2]), epoch)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(net.state_dict(), best_ckpt_path)

  net.load_state_dict(torch.load(best_ckpt_path))
  test_loss, test_acc = run_epoch(net, test_loader, criterion)
  logger.success(f"[regime {test_regime}, unseen] test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

  writer.add_hparams(
      {"model": model_name, "lr": lr, "batch_size": batch_size, "test_regime": test_regime},
      {"hparam/best_val_acc": best_val_acc, "hparam/cross_domain_test_acc": test_acc},
  )
  writer.close()

  return {"regime": test_regime, "best_val_acc": best_val_acc, "test_acc": test_acc}


@app.command()
def main(
    model: str = typer.Option(..., help="cnn | vit | cross_attention"),
    test_regime: int = typer.Option(
        None, help="Αν δοθεί, τρέχει ΜΟΝΟ αυτό το regime ως test. Αν όχι, "
                   "τρέχει leave-one-regime-out σε ΟΛΑ τα regimes (default)."
    ),
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 5e-5,
    features_path: Path = PROCESSED_DATA_DIR / "full_dataset.csv",
    tensors_path: Path = PROCESSED_DATA_DIR / "bearing_tensors.npy",
):
  """Τρέξε: python -m bearing_faults.modeling.train --model cnn
  (ή --model vit, --model cross_attention)

  Χωρίς --test-regime: leave-one-regime-out σε όλα τα regimes, με μέσο
  όρο/τυπική απόκλιση στο τέλος - πιο αξιόπιστο από ένα μόνο split.
  Με --test-regime N: τρέχει μόνο έναν γύρο, στο regime N.

  Προαπαιτούμενα: 1) dataset.py έχει ήδη τρέξει (full_dataset.csv, με
  στήλη source_file) 2) features.py έχει ήδη τρέξει (bearing_tensors.npy),
  ΙΔΙΑ window_size/stride και στα δύο.
  """
  df = pd.read_csv(features_path)
  tensors = np.load(tensors_path)
  assert len(df) == len(tensors), (
      f"CSV έχει {len(df)} γραμμές αλλά τα tensors {len(tensors)} - "
      "ξανάτρεξε python -m bearing_faults.features με ΙΔΙΑ window_size/"
      "stride όπως το dataset.py."
  )
  assert "source_file" in df.columns, (
      "Λείπει η στήλη 'source_file' από το CSV - ξανάτρεξε το "
      "ενημερωμένο dataset.py."
  )

  ss_mask = (df["state"] == "Steady-State").values
  df_ss = df[ss_mask].reset_index(drop=True)
  tensors_ss = tensors[ss_mask]

  regimes_to_run = [test_regime] if test_regime is not None else sorted(df_ss["regime_id"].unique().tolist())
  logger.info(f"Leave-one-regime-out πάνω σε regimes: {regimes_to_run}")

  results = []
  for i, regime in enumerate(regimes_to_run):
    results.append(
        train_one_regime(
            model_name=model, df_ss=df_ss, tensors_ss=tensors_ss, test_regime=regime,
            epochs=epochs, batch_size=batch_size, lr=lr, log_images=(i == 0),
        )
    )

  test_accs = np.array([r["test_acc"] for r in results])
  logger.success(
      f"\n=== {model}: leave-one-regime-out σε {len(results)} regime(s) ===\n"
      f"Test accuracy ανά regime: "
      + ", ".join(f"regime {r['regime']}={r['test_acc']:.4f}" for r in results)
      + f"\nMean ± std: {test_accs.mean():.4f} ± {test_accs.std():.4f}"
  )


if __name__ == "__main__":
  app()