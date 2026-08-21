"""Signal processing and multimodal 2D tensor generation."""
import loguru
import numpy as np
import pywt
import torch
import torch.nn as nn


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
        x: Tensor of shape [Batch, Channels=2, Freqs=64, Time=2000].
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