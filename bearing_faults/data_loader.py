"""PyTorch Dataset wrapper for bearing multimodal tensors."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BearingTensorDataset(Dataset):
  """Dataset providing:
  - scalogram: Tensor [2, 64, 64]
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
      max_time_shift: int = 8,
  ):
    """Args:
        tensors_2d: Numpy array of precomputed tensors [N, 2, 64, 64].
        df_metadata: Metadata DataFrame aligned with tensors_2d.
        label_col: Name of the class column.
        augment: If True, applies a random circular shift along the TIME axis
            (same shift for both channels), up to `max_time_shift` columns of
            the 64-column time axis. It only moves content in time, so it does
            not assume anything about the value range of the tensors (works on
            the dB-scaled [0, 1] input used in train.py). Use True for the
            training set only.
        max_time_shift: Maximum |shift| in columns.
    """
    if len(tensors_2d) != len(df_metadata):
      raise ValueError(
          f"tensors_2d has {len(tensors_2d)} rows but df_metadata has "
          f"{len(df_metadata)} rows — they must be aligned 1:1 in the "
          "same order."
      )

    self.augment = augment
    self.max_time_shift = max_time_shift
    self.tensors = torch.tensor(tensors_2d, dtype=torch.float32)
    self.df = df_metadata.reset_index(drop=True)

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

    # Prefer the numeric 'regime_id' column (produced directly by the
    # K-Means step in dataset.py) over parsing the display string
    # 'regime_cluster' ("Regime X") — parsing breaks silently if the
    # label format ever changes, whereas regime_id is the source of truth.
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
    x = self.tensors[idx]
    if self.augment and self.max_time_shift > 0:
      shift = int(torch.randint(-self.max_time_shift, self.max_time_shift + 1, (1,)).item())
      x = torch.roll(x, shifts=shift, dims=-1)
    return {
        "scalogram": x,
        "condition": self.conditions[idx],
        "label": self.labels[idx],
        "regime": self.regimes[idx],
    }