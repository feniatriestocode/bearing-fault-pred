"""Signal processing and multimodal 2D tensor generation.

CLI:
    python -m bearing_faults.features --window-size 1000 --stride 500

Builds bearing_tensors.npy ([N, 2, 64, 64], LINEAR power; the dB scaling is
done at training time), aligned 1:1 with full_dataset.csv from dataset.py.
"""
import glob
import os

from loguru import logger
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from tqdm import tqdm
import typer

from bearing_faults.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def compute_cwt_morlet(
    signal: np.ndarray,
    fs: int = 20000,
    f_min: float = 10.0,
    f_max: float = 8000.0,
    num_scales: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes CWT Morlet scalogram for a 1D window.

  Args:
      signal: 1D time series array of window samples.
      fs: Sampling frequency in Hz.
      f_min: Lower frequency boundary in Hz.
      f_max: Upper frequency boundary in Hz.
      num_scales: Number of wavelet scales / frequency bins.

  Returns:
      scalogram: 2D numpy array [num_scales, num_samples].
      frequencies: 1D numpy array [num_scales], Hz center of each row of
          the scalogram, in the SAME order as the scalogram rows. Required
          downstream to map any (possibly pooled) row back to a physical
          frequency for the physics-warped positional encoding.
  """
  dt = 1.0 / fs
  wavelet_name = "cmor1.5-1.0"
  frequencies = np.linspace(f_min, f_max, num_scales)
  scales = pywt.frequency2scale(wavelet_name, frequencies * dt)
  coefficients, _ = pywt.cwt(signal, scales, wavelet_name, sampling_period=dt)
  scalogram = np.abs(coefficients) ** 2
  return scalogram.astype(np.float32), frequencies.astype(np.float32)


class PhysicsPreservingDownsampler(nn.Module):
  """Differentiated channel-wise downsampler:
  - Channel 0 (Vibration): Adaptive Max-Pooling preserves transient shock
    peaks.
  - Channel 1 (Current): Adaptive Avg-Pooling preserves continuous energy
    density.

  Vibration and current are computed with DIFFERENT f_min/f_max ranges
  (see dataset.py: vib ~50-8000Hz, current ~10-1000Hz), so each channel
  needs its own frequency-bin bookkeeping — a single shared freq axis
  would be wrong for one of the two channels.
  """

  def __init__(self, output_size: tuple[int, int] = (64, 64)):
    super().__init__()
    self.output_size = output_size
    self.max_pool = nn.AdaptiveMaxPool2d(output_size)
    self.avg_pool = nn.AdaptiveAvgPool2d(output_size)

  @staticmethod
  def _pool_freq_axis(freq_bins_hz: np.ndarray, out_freq_len: int) -> np.ndarray:
    """Downsamples a 1D array of frequency-bin centers (Hz) to match the
    pooled tensor's frequency axis length, via linear interpolation over
    bin index. This is an approximation (max-pool is non-linear so the
    'true' representative frequency of a pooled bin is ill-defined), but
    it is the standard, defensible choice: it gives each pooled row the
    Hz value at the center of the original bins it was built from.
    """
    if len(freq_bins_hz) == out_freq_len:
      return freq_bins_hz.astype(np.float32)
    src_idx = np.arange(len(freq_bins_hz))
    tgt_idx = np.linspace(0, len(freq_bins_hz) - 1, out_freq_len)
    return np.interp(tgt_idx, src_idx, freq_bins_hz).astype(np.float32)

  def forward(
      self,
      x: torch.Tensor,
      vib_freq_bins_hz: np.ndarray,
      cur_freq_bins_hz: np.ndarray,
  ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Args:
        x: Tensor of shape [Batch, Channels=2, Freqs=64, Time=window_size].
           Channel 0 = vibration, Channel 1 = current.
        vib_freq_bins_hz: 1D array, Hz centers of x[:,0]'s frequency axis
            BEFORE pooling (i.e. the `frequencies` returned by
            compute_cwt_morlet for the vibration signal).
        cur_freq_bins_hz: same, for the current signal (x[:,1]).

    Returns:
        pooled: Tensor of shape [Batch, Channels=2, Height, Width].
        pooled_freq_bins: dict with 'vibration' and 'current' keys, each a
            1D numpy array of length `Height` giving the Hz center of each
            row AFTER pooling. Use this (not a re-derived linspace) when
            computing distance-to-BPFO/BPFI for the physics-warped PE.
    """
    vib_ds = self.max_pool(x[:, 0:1])
    cur_ds = self.avg_pool(x[:, 1:2])
    pooled = torch.cat([vib_ds, cur_ds], dim=1)

    out_h = self.output_size[0]
    pooled_freq_bins = {
        "vibration": self._pool_freq_axis(vib_freq_bins_hz, out_h),
        "current": self._pool_freq_axis(cur_freq_bins_hz, out_h),
    }
    return pooled, pooled_freq_bins


# ----------------------------------------------------------------------
# CLI: builds bearing_tensors.npy, aligned 1:1 with full_dataset.csv
# (same file/window order as dataset.py: categories in the given order,
# sorted() files, windows 0..n-1)
# ----------------------------------------------------------------------
@app.command()
def main(
    categories: list[str] = ["Damage", "Healthy", "Inner", "Outer"],
    window_size: int = 1000,
    stride: int = 500,
    fs: int = 20000,
    output_size: int = 64,
):
  """Re-reads the raw .xlsx files and builds the [N, 2, 64, 64] scalogram
  tensors, in EXACTLY the same file/window order as dataset.py, so that they
  match full_dataset.csv row by row.

  IMPORTANT: window_size/stride MUST be the same as the ones passed to
  `python -m bearing_faults.dataset`, otherwise the CSV and the tensors have a
  different number of rows. Run AFTER dataset.py.
  """
  downsampler = PhysicsPreservingDownsampler(output_size=(output_size, output_size))

  tensors = []
  keys = []  # source_file of every window, in order, to verify the alignment with the CSV
  freq_bins_vib, freq_bins_cur = None, None

  for cat in categories:
    cat_path = RAW_DATA_DIR / cat
    xlsx_files = sorted(glob.glob(str(cat_path / "*.xlsx")))
    logger.info(f"'{cat}': {len(xlsx_files)} files")

    for f in tqdm(xlsx_files, desc=cat):
      vib_df = pd.read_excel(f, sheet_name="Vibration").drop(columns=["Time"])
      i_df = pd.read_excel(f, sheet_name="Current").drop(columns=["Time"])
      assert len(vib_df) == len(i_df), f"Vibration/Current length differs: {f}"
      n_windows = max(0, (len(vib_df) - window_size) // stride + 1)

      for w in range(n_windows):
        s = w * stride
        e = s + window_size
        vib_win = vib_df["Vibration"].iloc[s:e].values.astype(np.float32)
        cur_win = i_df.iloc[s:e, 0].values.astype(np.float32)  # Current1 (phase 1)

        vib_scalo, vib_freqs = compute_cwt_morlet(
            vib_win, fs=fs, f_min=50, f_max=8000, num_scales=64
        )
        cur_scalo, cur_freqs = compute_cwt_morlet(
            cur_win, fs=fs, f_min=10, f_max=1000, num_scales=64
        )

        x = torch.tensor(np.stack([vib_scalo, cur_scalo]))[None]  # [1, 2, 64, window_size]
        with torch.no_grad():
          pooled, freq_bins = downsampler(x, vib_freqs, cur_freqs)
        tensors.append(pooled[0].numpy())  # [2, 64, 64], LINEAR power
        keys.append(os.path.basename(f))

        if freq_bins_vib is None:
          freq_bins_vib = freq_bins["vibration"]
          freq_bins_cur = freq_bins["current"]

  if not tensors:
    raise FileNotFoundError(f"No windows built. Check {RAW_DATA_DIR} and --window-size.")
  tensors_2d = np.stack(tensors)  # [N, 2, 64, 64]

  # Alignment check against full_dataset.csv (if it exists)
  csv_path = PROCESSED_DATA_DIR / "full_dataset.csv"
  if csv_path.exists():
    csv_keys = pd.read_csv(csv_path, usecols=["source_file"])["source_file"].tolist()
    if csv_keys != keys:
      raise RuntimeError(
          f"Tensors ({len(keys)} rows) are NOT aligned with {csv_path.name}"
          f" ({len(csv_keys)} rows). Re-run dataset.py and features.py with the"
          " same --window-size / --stride / --categories. Nothing was saved."
      )
    logger.success(f"Alignment with {csv_path.name} verified ({len(keys)} rows).")
  else:
    logger.warning(f"{csv_path} not found: alignment NOT verified. Run dataset.py first.")

  os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
  np.save(PROCESSED_DATA_DIR / "bearing_tensors.npy", tensors_2d)
  np.save(PROCESSED_DATA_DIR / "freq_bins_vib.npy", freq_bins_vib)
  np.save(PROCESSED_DATA_DIR / "freq_bins_cur.npy", freq_bins_cur)
  logger.success(f"Built {len(keys)} tensors, shape {tensors_2d.shape}")


if __name__ == "__main__":
  app()