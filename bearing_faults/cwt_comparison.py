"""Σύγκριση κλάσεων, ΚΑΙ για vibration ΚΑΙ για current:
1. Mean scalogram πάνω σε ΟΛΑ τα windows μιας κλάσης ΠΟΥ ΑΝΗΚΟΥΝ ΣΤΟ ΙΔΙΟ
   REGIME (σταθερή ταχύτητα/φορτίο) - όχι σε όλη την κλάση αδιακρίτως,
   γιατί οι fault frequencies μετατοπίζονται με την ταχύτητα.
2. Difference plot (κλάση - Healthy), ΜΕΣΑ ΣΤΟ ΙΔΙΟ REGIME.
3. Outlier diagnostic: per-window energy πριν το average, να δούμε αν το
   mean κυριαρχείται από λίγα ακραία windows.

Κάθε κανάλι (vibration, current) έχει το δικό του εύρος συχνοτήτων -
BPFO/BPFI για vibration ζουν ψηλά (50-8000Hz), sidebands για current ζουν
κοντά στη line frequency (10-150Hz).
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
TARGET_REGIME = 2
OUT_DIR = '../reports/scalograms'
os.makedirs(OUT_DIR, exist_ok=True)

# Ανά κανάλι: (sheet_name, f_min, f_max)
CHANNEL_CONFIG = {
    'vibration': {'sheet': 'Vibration', 'f_min': 50, 'f_max': 8000},
    'current': {'sheet': 'Current', 'f_min': 10, 'f_max': 150},
}

df = pd.read_csv('../data/processed/full_dataset.csv')


def _extract_window(sheet_df: pd.DataFrame, channel: str, s: int, e: int) -> np.ndarray:
  """Vibration sheet έχει 1 στήλη ('Vibration'), Current sheet έχει 3
  (Current1/2/3) - παίρνουμε πάντα την πρώτη διαθέσιμη στήλη σήματος."""
  if channel == 'vibration':
    return sheet_df['Vibration'].iloc[s:e].values.astype(np.float32)
  return sheet_df.iloc[s:e, 0].values.astype(np.float32)  # Current1


def mean_scalogram_for_regime(
    cat: str, channel: str, regime_id: int, max_windows: int = 200
) -> tuple[np.ndarray, np.ndarray, int]:
  """Μέσος όρος |CWT|^2 πάνω σε ΟΛΑ τα steady-state windows της κλάσης
  `cat` στο συγκεκριμένο `regime_id`, για το δοσμένο `channel`."""
  cfg = CHANNEL_CONFIG[channel]
  cat_path = os.path.join(ROOT_DIR, cat)
  xlsx_files = sorted(glob.glob(os.path.join(cat_path, '*.xlsx')))

  scalos, freqs = [], None
  global_window_idx = 0
  cat_rows = df[df['fault_class'] == cat]

  for f in xlsx_files:
    sheet_df = pd.read_excel(f, sheet_name=cfg['sheet']).drop(columns=['Time'])
    n_windows = len(sheet_df) // WINDOW_SIZE

    for w in range(n_windows):
      row = cat_rows.iloc[global_window_idx] if global_window_idx < len(cat_rows) else None
      global_window_idx += 1
      if row is None or row['regime_id'] != regime_id or row['state'] != 'Steady-State':
        continue
      if len(scalos) >= max_windows:
        break

      s, e = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
      win = _extract_window(sheet_df, channel, s, e)
      scalo, freqs = compute_cwt_morlet(win, fs=FS, f_min=cfg['f_min'], f_max=cfg['f_max'], num_scales=64)
      scalos.append(scalo)

    if len(scalos) >= max_windows:
      break

  if not scalos:
    raise ValueError(f'Καμία steady-state window για {cat}/{channel} στο regime {regime_id}')
  return np.mean(scalos, axis=0), freqs, len(scalos)


def per_window_energies_for_regime(
    cat: str, channel: str, regime_id: int, max_windows: int = 200
) -> list:
  """Per-window total energy (πριν το average) - για τον outlier έλεγχο."""
  cfg = CHANNEL_CONFIG[channel]
  cat_path = os.path.join(ROOT_DIR, cat)
  xlsx_files = sorted(glob.glob(os.path.join(cat_path, '*.xlsx')))

  energies = []
  global_window_idx = 0
  cat_rows = df[df['fault_class'] == cat]

  for f in xlsx_files:
    sheet_df = pd.read_excel(f, sheet_name=cfg['sheet']).drop(columns=['Time'])
    n_windows = len(sheet_df) // WINDOW_SIZE

    for w in range(n_windows):
      row = cat_rows.iloc[global_window_idx] if global_window_idx < len(cat_rows) else None
      global_window_idx += 1
      if row is None or row['regime_id'] != regime_id or row['state'] != 'Steady-State':
        continue
      if len(energies) >= max_windows:
        break

      s, e = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
      win = _extract_window(sheet_df, channel, s, e)
      scalo, _ = compute_cwt_morlet(win, fs=FS, f_min=cfg['f_min'], f_max=cfg['f_max'], num_scales=64)
      energies.append(scalo.sum())

    if len(energies) >= max_windows:
      break

  return energies


def run_for_channel(channel: str) -> None:
  print(f'\n========== {channel.upper()} ==========')
  n_used_map, mean_scalos = {}, {}
  freqs_ref = None

  for cat in CATEGORIES:
    mean_scalos[cat], freqs_ref, n_used_map[cat] = mean_scalogram_for_regime(cat, channel, TARGET_REGIME)
    print(f'{cat} (regime {TARGET_REGIME}): averaged {n_used_map[cat]} steady-state windows')

  # --- Plot 1: mean scalogram ανά κλάση ---
  fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
  vmax = max(m.max() for m in mean_scalos.values())
  for ax, cat in zip(axes, CATEGORIES):
    im = ax.imshow(
        np.log1p(mean_scalos[cat]), aspect='auto', origin='lower',
        extent=[0, WINDOW_SIZE / FS, freqs_ref.min(), freqs_ref.max()],
        cmap='viridis', vmin=0, vmax=np.log1p(vmax),
    )
    ax.set_title(f'{cat} (n={n_used_map[cat]}, regime {TARGET_REGIME})')
    ax.set_xlabel('Time (s)')
  axes[0].set_ylabel('Frequency (Hz)')
  fig.colorbar(im, ax=axes, label='log(1 + |CWT|²)', shrink=0.8)
  plt.savefig(os.path.join(OUT_DIR, f'{channel}_mean_regime{TARGET_REGIME}_by_class.png'), dpi=130, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {channel}_mean_regime{TARGET_REGIME}_by_class.png')

  # --- Plot 2: difference vs Healthy ---
  healthy_mean = mean_scalos['Healthy']
  fault_classes = [c for c in CATEGORIES if c != 'Healthy']
  fig, axes = plt.subplots(1, len(fault_classes), figsize=(15, 4.5), sharey=True)
  diffs = [mean_scalos[c] - healthy_mean for c in fault_classes]
  abs_max = max(np.abs(d).max() for d in diffs)
  for ax, cat, diff in zip(axes, fault_classes, diffs):
    im = ax.imshow(
        diff, aspect='auto', origin='lower',
        extent=[0, WINDOW_SIZE / FS, freqs_ref.min(), freqs_ref.max()],
        cmap='RdBu_r', vmin=-abs_max, vmax=abs_max,
    )
    ax.set_title(f'{cat} − Healthy')
    ax.set_xlabel('Time (s)')
  axes[0].set_ylabel('Frequency (Hz)')
  fig.colorbar(im, ax=axes, label='Difference in |CWT|²', shrink=0.8)
  plt.savefig(os.path.join(OUT_DIR, f'{channel}_diff_vs_healthy.png'), dpi=130, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved {channel}_diff_vs_healthy.png')

  # --- Plot 3: outlier diagnostic ---
  print(f'--- Per-window total energy ({channel}, πριν το average) ---')
  energy_data = {cat: per_window_energies_for_regime(cat, channel, TARGET_REGIME) for cat in CATEGORIES}
  for cat, vals in energy_data.items():
    vals = np.array(vals)
    print(f'{cat}: n={len(vals)}, median={np.median(vals):.3e}, mean={vals.mean():.3e}, '
          f'max={vals.max():.3e} (max/median ratio={vals.max() / np.median(vals):.1f}x)')

  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.boxplot([energy_data[c] for c in CATEGORIES], tick_labels=CATEGORIES, showfliers=True)
  ax.set_ylabel('Per-window total |CWT|² energy')
  ax.set_title(f'{channel}: κατανομή ενέργειας ανά κλάση, regime {TARGET_REGIME}')
  plt.tight_layout()
  plt.savefig(os.path.join(OUT_DIR, f'{channel}_energy_outlier_check.png'), dpi=130)
  plt.close(fig)
  print(f'Saved {channel}_energy_outlier_check.png')


if __name__ == '__main__':
  for channel in ['vibration', 'current']:
    run_for_channel(channel)