"""Saves one representative RAW CWT-Morlet scalogram (vibration + current)

per fault class in dB scale using absolute paths from config.
"""

import glob
import os
from pathlib import Path
from bearing_faults.config import RAW_DATA_DIR
from features import compute_cwt_morlet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = ['Damage', 'Healthy', 'Inner', 'Outer']
WINDOW_SIZE = 1000  # 1000 samples ώστε να ταιριάζει με το pipeline του μοντέλου
FS = 20000

# Αποθήκευση στο reports/figures/scalograms του project
OUT_DIR = Path(RAW_DATA_DIR).resolve().parent.parent / 'reports' / 'scalograms'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_db(scalo, dynamic_range_db=50.0):
  scalo_db = 10.0 * np.log10(np.maximum(scalo, 1e-12))
  vmax = scalo_db.max()
  vmin = vmax - dynamic_range_db
  return scalo_db, vmin, vmax


for cat in CATEGORIES:
  cat_path = Path(RAW_DATA_DIR) / cat
  xlsx_files = sorted(glob.glob(str(cat_path / '*.xlsx')))

  # Ασφαλής έλεγχος πριν το index [0]
  if not xlsx_files:
    print(f"Δεν βρέθηκαν .xlsx αρχεία στον φάκελο: {cat_path}")
    continue

  first_file = xlsx_files[0]
  print(f"Επεξεργασία {cat} από: {os.path.basename(first_file)}")

  vib_df = pd.read_excel(first_file, sheet_name='Vibration').drop(
      columns=['Time']
  )
  i_df = pd.read_excel(first_file, sheet_name='Current').drop(columns=['Time'])

  vib_win = vib_df['Vibration'].iloc[0:WINDOW_SIZE].values.astype(np.float32)
  cur_win = i_df.iloc[0:WINDOW_SIZE, 0].values.astype(np.float32)

  vib_scalo, vib_freqs = compute_cwt_morlet(
      vib_win, fs=FS, f_min=50, f_max=8000, num_scales=64
  )
  cur_scalo, cur_freqs = compute_cwt_morlet(
      cur_win, fs=FS, f_min=10, f_max=1000, num_scales=64
  )

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
  plots = [
      (axes[0], vib_scalo, vib_freqs, 'Vibration', 'viridis'),
      (axes[1], cur_scalo, cur_freqs, 'Current (Phase 1)', 'magma'),
  ]

  for ax, scalo, freqs, name, cmap in plots:
    scalo_db, vmin, vmax = to_db(scalo, dynamic_range_db=50.0)

    # Έλεγχος διάταξης αξόνων
    if freqs[0] > freqs[-1]:
      scalo_db = np.flipud(scalo_db)
      f_low, f_high = freqs[-1], freqs[0]
    else:
      f_low, f_high = freqs[0], freqs[-1]

    time_extent = WINDOW_SIZE / FS

    im = ax.imshow(
        scalo_db,
        aspect='auto',
        origin='lower',
        extent=[0, time_extent, f_low, f_high],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(f'{cat} — {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)', fontsize=10)

  plt.tight_layout()
  out_path = OUT_DIR / f'{cat}_raw_scalogram.png'
  plt.savefig(out_path, dpi=140)
  plt.close(fig)
  print(f'Αποθηκεύτηκε: {out_path}')