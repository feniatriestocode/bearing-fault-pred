"""PyTorch Dataset wrapper for bearing multimodal tensors."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def standardize_per_instance(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """Z-score standardization, ΑΝΑ ΔΕΙΓΜΑ ΚΑΙ ΑΝΑ ΚΑΝΑΛΙ ξεχωριστά (όχι με
  στατιστικά υπολογισμένα στο training set - είναι local operation, ίδια
  λογική εφαρμόζεται σε train/val/test χωρίς leakage).

  Αφαιρεί το απόλυτο πλάτος (amplitude) κάθε δείγματος, αναγκάζοντας το
  μοντέλο να μάθει το ΣΧΗΜΑ του φάσματος αντί για το συνολικό scale -
  αυτό ακριβώς χτυπάει το amplitude-shortcut που βρήκαμε (vibration total
  energy ~7-9x διαφορετικό Healthy vs faulty, ανεξάρτητα από πραγματικό
  διαγνωστικό pattern).

  x: [C, H, W] (ένα δείγμα, C κανάλια - εδώ 2: vibration, current)
  """
  c = x.shape[0]
  x_flat = x.view(c, -1)
  mean = x_flat.mean(dim=1, keepdim=True)
  std = x_flat.std(dim=1, keepdim=True)
  x_norm = (x_flat - mean) / (std + eps)
  return x_norm.view_as(x)


def augment_scalogram(
    x: torch.Tensor,
    noise_std: float = 0.03,
    max_time_mask: int = 8,
    max_freq_mask: int = 8,
    p_mask: float = 0.5,
) -> torch.Tensor:
  """Augmentation ΜΟΝΟ για training split:
  - Additive Gaussian noise (μικρό, σχετικό με το ήδη standardized scale).
  - SpecAugment-style time/frequency masking: τυχαία "σβήνει" μια συνεχή
    λωρίδα στον χρόνο ή στη συχνότητα, ανά κανάλι ξεχωριστά - αναγκάζει το
    μοντέλο να μη βασίζεται σε ένα μόνο "εύκολο" σημείο του scalogram.

  x: [C, H, W], ήδη standardized.
  """
  x = x.clone()
  c, h, w = x.shape

  # Additive noise
  x = x + torch.randn_like(x) * noise_std

  # Frequency masking (κατά μήκος του H άξονα)
  if torch.rand(1).item() < p_mask:
    for ch in range(c):
      f_width = torch.randint(1, max_freq_mask + 1, (1,)).item()
      f_start = torch.randint(0, max(1, h - f_width), (1,)).item()
      x[ch, f_start:f_start + f_width, :] = 0.0

  # Time masking (κατά μήκος του W άξονα)
  if torch.rand(1).item() < p_mask:
    for ch in range(c):
      t_width = torch.randint(1, max_time_mask + 1, (1,)).item()
      t_start = torch.randint(0, max(1, w - t_width), (1,)).item()
      x[ch, :, t_start:t_start + t_width] = 0.0

  return x


class BearingTensorDataset(Dataset):
  """Dataset providing:
  - scalogram: Tensor [2, 64, 64] (standardized, +augmented αν augment=True)
  - condition: Tensor [2] -> [speed_mean, torque_mean]
  - label: LongTensor (0: Healthy, 1: Inner, 2: Outer, 3: Damage)
  - regime: LongTensor indicating K-Means cluster ID

  NOTE: df_metadata is expected to already be filtered to the rows you
  actually want to train/eval on (e.g. state == 'Steady-State') and to be
  row-aligned with tensors_2d (same order, same length). This class does
  not filter or reorder anything for you.
  """

  def __init__(
      self,
      tensors_2d: np.ndarray,
      df_metadata: pd.DataFrame,
      label_col: str = "fault_class",
      augment: bool = False,
  ):
    """Args:
        tensors_2d: Numpy array of precomputed tensors [N, 2, 64, 64].
        df_metadata: Metadata DataFrame aligned with tensors_2d.
        label_col: Name of the class column.
        augment: αν True, εφαρμόζει augment_scalogram() σε κάθε δείγμα -
            βάλε True ΜΟΝΟ για το training split, False για val/test
            (θες σταθερή, μη-τυχαία αξιολόγηση εκεί).
    """
    if len(tensors_2d) != len(df_metadata):
      raise ValueError(
          f"tensors_2d has {len(tensors_2d)} rows but df_metadata has "
          f"{len(df_metadata)} rows — they must be aligned 1:1 in the "
          "same order."
      )

    self.tensors = torch.tensor(tensors_2d, dtype=torch.float32)
    self.df = df_metadata.reset_index(drop=True)
    self.augment = augment

    label_map = {"Healthy": 0, "Inner": 1, "Outer": 2, "Damage": 3}
    label_series = self.df[label_col]
    if pd.api.types.is_string_dtype(label_series) or label_series.dtype == object:
      unmapped = set(label_series.unique()) - set(label_map.keys())
      if unmapped:
        raise ValueError(
            f"Unrecognized values in '{label_col}': {unmapped}. "
            f"Expected one of {list(label_map.keys())}."
        )
      self.labels = torch.tensor(
          label_series.map(label_map).values, dtype=torch.long
      )
    else:
      self.labels = torch.tensor(label_series.values, dtype=torch.long)

    self.conditions = torch.tensor(
        self.df[["speed_mean", "torque_mean"]].values, dtype=torch.float32
    )

    if "regime_id" in self.df.columns:
      self.regimes = torch.tensor(
          self.df["regime_id"].values, dtype=torch.long
      )
    elif "regime_cluster" in self.df.columns:
      regimes = (
          self.df["regime_cluster"].str.replace("Regime ", "", regex=False).astype(int)
      )
      self.regimes = torch.tensor(regimes.values, dtype=torch.long)
    else:
      self.regimes = torch.zeros(len(self.df), dtype=torch.long)

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
    scalogram = standardize_per_instance(self.tensors[idx])
    if self.augment:
      scalogram = augment_scalogram(scalogram)

    return {
        "scalogram": scalogram,
        "condition": self.conditions[idx],
        "label": self.labels[idx],
        "regime": self.regimes[idx],
    }