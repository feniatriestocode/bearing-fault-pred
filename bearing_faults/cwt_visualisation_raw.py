"""Saves one representative RAW CWT-Morlet scalogram (vibration + current)
per fault class to ../reports/scalograms, computed directly from the .xlsx
files — no dependency on build_tensors.py or any pooled/downsampled tensor.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import compute_cwt_morlet

ROOT_DIR = '../../data'
CATEGORIES = ['Damage', 'Healthy', 'Inner', 'Outer']
WINDOW_SIZE = 2000
FS = 20000
OUT_DIR = '../reports/scalograms'
os.makedirs(OUT_DIR, exist_ok=True)

for cat in CATEGORIES:
  cat_path = os.path.join(ROOT_DIR, cat)
  xlsx_files = sorted(glob.glob(os.path.join(cat_path, '*.xlsx')))
  first_file = xlsx_files[0]  # ένα representative αρχείο ανά κλάση αρκεί για οπτικό έλεγχο

  vib_df = pd.read_excel(first_file, sheet_name='Vibration').drop(columns=['Time'])
  i_df = pd.read_excel(first_file, sheet_name='Current').drop(columns=['Time'])

  vib_win = vib_df['Vibration'].iloc[0:WINDOW_SIZE].values.astype(np.float32)
  cur_win = i_df.iloc[0:WINDOW_SIZE, 0].values.astype(np.float32)  # Current1

  vib_scalo, vib_freqs = compute_cwt_morlet(vib_win, fs=FS, f_min=50, f_max=8000, num_scales=64)
  cur_scalo, cur_freqs = compute_cwt_morlet(cur_win, fs=FS, f_min=10, f_max=1000, num_scales=64)

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
  for ax, scalo, freqs, name in [
      (axes[0], vib_scalo, vib_freqs, 'Vibration'),
      (axes[1], cur_scalo, cur_freqs, 'Current'),
  ]:
    im = ax.imshow(
        scalo, aspect='auto', origin='lower',
        extent=[0, scalo.shape[-1] / FS, freqs.min(), freqs.max()],
        cmap='viridis',
    )
    ax.set_title(f'{cat} — {name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=ax, label='|CWT coefficient|²')

  plt.tight_layout()
  out_path = os.path.join(OUT_DIR, f'{cat}_raw_scalogram.png')
  plt.savefig(out_path, dpi=130)
  plt.close(fig)
  print(f'Saved {out_path} (from {os.path.basename(first_file)})')