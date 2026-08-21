import os
import glob
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def get_cwt_energy(signal, fs=20000, f_min=50, f_max=5000, num_scales=32):
    """Υπολογίζει τη συνολική ενέργεια CWT-Morlet ενός σήματος παραθύρου"""
    dt = 1.0 / fs
    wavelet = 'cmor1.5-1.0'
    freqs = np.linspace(f_min, f_max, num_scales)
    scales = pywt.frequency2scale(wavelet, freqs * dt)
    coefs, _ = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
    return np.sum(np.abs(coefs)**2)

def process_bearing_file(file_path, label, load_level, window_size=2000, fs=20000, smooth_window=50):
    """
    smooth_window: size of the rolling mean (in samples) applied
    ONLY to speed/torque signals before calculating the standard deviation
    for each window. It removes encoder/quantization noise so the standard
    deviation reflects actual changes in operating condition rather than
    sampling noise. It is not applied to current/vibration/voltage, which
    require their raw, high-frequency signals for the CWT.
    """
    v_df = pd.read_excel(file_path, sheet_name='Voltage').drop(columns=['Time'])
    i_df = pd.read_excel(file_path, sheet_name='Current').drop(columns=['Time'])
    t_df = pd.read_excel(file_path, sheet_name='Torque').drop(columns=['Time'])
    s_df = pd.read_excel(file_path, sheet_name='Speed').drop(columns=['Time'])
    vib_df = pd.read_excel(file_path, sheet_name='Vibration').drop(columns=['Time'])

    # Smoothing

    s_smooth = s_df['Speed'].rolling(window=smooth_window, center=True, min_periods=1).mean()
    t_smooth = t_df['Torque'].rolling(window=smooth_window, center=True, min_periods=1).mean()

    n_samples = len(t_df)
    n_windows = n_samples // window_size
    records = []

    for w in range(n_windows):
        idx_start = w * window_size
        idx_end = idx_start + window_size

        t_win = t_smooth.iloc[idx_start:idx_end].values
        s_win = s_smooth.iloc[idx_start:idx_end].values
        t_win_raw = t_df['Torque'].iloc[idx_start:idx_end].values
        s_win_raw = s_df['Speed'].iloc[idx_start:idx_end].values
        v_win = v_df.iloc[idx_start:idx_end].values          # Voltage1/2/3
        i_win = i_df.iloc[idx_start:idx_end].values          # Current1/2/3
        vib_win = vib_df['Vibration'].iloc[idx_start:idx_end].values

        # Μηχανικές παράμετροι (πάνω σε smoothed σήμα)
        mean_torque = np.mean(t_win)
        std_torque = np.std(t_win)
        mean_speed = np.mean(s_win)
        std_speed = np.std(s_win)

        # Ίδιο std πάνω στο raw (χωρίς smoothing) σήμα, μόνο για σύγκριση/διάγραμμα
        std_torque_raw = np.std(t_win_raw)
        std_speed_raw = np.std(s_win_raw)

        # Ηλεκτρικές παράμετροι (3-Phase RMS) — raw σήμα
        i_rms = np.sqrt(np.mean(i_win**2))
        v_rms = np.sqrt(np.mean(v_win**2))

        # Στατιστικά κραδασμών — raw σήμα
        a_rms = np.sqrt(np.mean(vib_win**2))
        a_peak = np.max(np.abs(vib_win))
        crest_factor = a_peak / (a_rms + 1e-8)

        # CWT-Morlet Energies — raw σήμα
        cwt_energy_vib = get_cwt_energy(vib_win, fs=fs, f_min=50, f_max=8000)
        cwt_energy_cur = get_cwt_energy(i_win[:, 0], fs=fs, f_min=10, f_max=1000)

        records.append({
            'fault_class': label,
            'nominal_load': load_level,
            'torque_mean': mean_torque,
            'torque_std': std_torque,
            'torque_std_raw': std_torque_raw,
            'speed_mean': mean_speed,
            'speed_std': std_speed,
            'speed_std_raw': std_speed_raw,
            'i_rms': i_rms,
            'v_rms': v_rms,
            'a_rms': a_rms,
            'a_peak': a_peak,
            'crest_factor': crest_factor,
            'cwt_energy_vib': cwt_energy_vib,
            'cwt_energy_cur': cwt_energy_cur
        })

    return pd.DataFrame(records)

root_dir = '../../data'  # Ο φάκελος όπου βρίσκονται οι υποφάκελοι Damage, Healthy, Inner, Outer
logger.info(f"Scanning root directory: {root_dir}")
categories = ['Damage', 'Healthy', 'Inner', 'Outer']
all_dfs = []

for cat in categories:
    cat_path = os.path.join(root_dir, cat)
    xlsx_files = glob.glob(os.path.join(cat_path, '*.xlsx'))
    logger.info(f"Processing category '{cat}' with {len(xlsx_files)} files.")
    for f in xlsx_files:
        filename = os.path.basename(f)
        load_tag = filename.split('_')[-1].replace('.xlsx', '') + '%'
        df_file = process_bearing_file(f, label=cat, load_level=load_tag)
        all_dfs.append(df_file)

full_dataset = pd.concat(all_dfs, ignore_index=True)
# save full_dataset in data/processed
full_dataset.to_csv('../data/processed/full_dataset.csv', index=False)   


logger.info(f"Total windows: {len(full_dataset)}")

logger.info(
    f"Speed std -> Min: {full_dataset['speed_std'].min():.2f}, Max:"
    f" {full_dataset['speed_std'].max():.2f}, Mean:"
    f" {full_dataset['speed_std'].mean():.2f}"
)
logger.info(
    f"Torque std -> Min: {full_dataset['torque_std'].min():.2f}, Max:"
    f" {full_dataset['torque_std'].max():.2f}, Mean:"
    f" {full_dataset['torque_std'].mean():.2f}"
)

# Δες την πραγματική κατανομή ΠΡΙΝ διαλέξεις threshold - μην μαντεύεις τιμή
# Before/after smoothing σε ίδια κλίμακα ανά μεταβλητή, για άμεση σύγκριση
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
full_dataset['speed_std_raw'].hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Speed std (before smoothing)')
full_dataset['speed_std'].hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Speed std (after smoothing)')
full_dataset['torque_std_raw'].hist(bins=50, ax=axes[1, 0])
axes[1, 0].set_title('Torque std (before smoothing)')
full_dataset['torque_std'].hist(bins=50, ax=axes[1, 1])
axes[1, 1].set_title('Torque std (after smoothing)')
plt.tight_layout()
plt.savefig('../reports/figures/std_distributions.png', dpi=120)
plt.show()

logger.info(
    "Δες το std_distributions.png (ή το παράθυρο που άνοιξε) πριν ορίσεις "
    "τα thresholds steady-state παρακάτω. Πρότεινε π.χ. 90th percentile "
    "αντί για αυθαίρετη τιμή."
)

speed_std_threshold = full_dataset['speed_std'].quantile(0.90)
torque_std_threshold = full_dataset['torque_std'].quantile(0.90)

logger.info(
    f"Use of 90th percentile thresholds -> speed_std < {speed_std_threshold:.2f}, "
    f"torque_std < {torque_std_threshold:.2f}"
)

is_ss = (full_dataset["speed_std"] < speed_std_threshold) & (
    full_dataset["torque_std"] < torque_std_threshold
)

full_dataset["state"] = np.where(is_ss, "Steady-State", "Transient")
df_ss = full_dataset[full_dataset["state"] == "Steady-State"].copy()

logger.info(f"Steady-State δείγματα: {len(df_ss)} / {len(full_dataset)}")

if len(df_ss) == 0:
    logger.warning(
        "The filter was too strict! Using all data for clustering."
    )
    df_ss = full_dataset.copy()

from sklearn.metrics import silhouette_score

scaler = StandardScaler()
scaled_op_points = scaler.fit_transform(df_ss[["torque_mean", "speed_mean"]])

K_range = range(2, 9)
inertias, sil_scores = [], []
for k_try in K_range:
    km_try = KMeans(n_clusters=k_try, random_state=42, n_init=10).fit(scaled_op_points)
    inertias.append(km_try.inertia_)
    sil_scores.append(silhouette_score(scaled_op_points, km_try.labels_))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(K_range), inertias, marker='o')
axes[0].set_xlabel('k (# clusters)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow method')
axes[1].plot(list(K_range), sil_scores, marker='o', color='darkorange')
axes[1].set_xlabel('k (# clusters)')
axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette score ανά k')
plt.tight_layout()
plt.savefig('../reports/figures/kmeans_k_selection.png', dpi=120)
plt.show()

logger.info(
    "kmeans_k_selection.png: max silhouette score -> "
    f"k={list(K_range)[int(np.argmax(sil_scores))]} (elbow: inspect visually"
    " στο αριστερό plot)."
)

k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_ss["regime_cluster"] = kmeans.fit_predict(scaled_op_points)
df_ss["regime_cluster"] = "Regime " + df_ss["regime_cluster"].astype(str)

logger.success("Το 2D K-Means clustering completed successfully!")


centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(7.5, 6.5))
sns.scatterplot(
    data=df_ss, x='torque_mean', y='speed_mean',
    hue='regime_cluster', palette='tab10', s=35, alpha=0.6
)
plt.scatter(
    centers_original[:, 0], centers_original[:, 1],
    marker='X', s=220, c='black', edgecolors='white', linewidths=1.5,
    label='Cluster centers', zorder=5
)
plt.xlabel('Torque (mean, per window)')
plt.ylabel('Speed (mean, per window)')
plt.title(f'K-Means (k={k}) (torque, speed)')
plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left')
plt.tight_layout()
plt.savefig('../reports/figures/kmeans_regime_clusters.png', dpi=120, bbox_inches='tight')
plt.show()

sns.set_theme(style="whitegrid", palette="tab10")

g = sns.FacetGrid(
    df_ss,
    col='regime_cluster',
    col_wrap=3,
    height=4.5,
    aspect=1.2,
    sharex=False,
    sharey=False
)

g.map_dataframe(
    sns.scatterplot,
    x='i_rms',
    y='a_rms',
    hue='fault_class',
    style='fault_class',
    s=70,
    alpha=0.85
)

g.set_axis_labels('Stator Current $I_{RMS}$ (A)', 'Vibration Acceleration $a_{RMS}$ (g)')
g.add_legend(title='Bearing Fault Class', bbox_to_anchor=(1.02, 0.5), loc='center left')
g.figure.subplots_adjust(top=0.9, right=0.88)
g.figure.suptitle(
    'Fault Clustering across Joint (Torque, Speed) Operating Regimes',
    fontsize=15, weight='bold'
)

plt.savefig('../reports/figures/fault_clustering_facetgrid.png', dpi=120, bbox_inches='tight')
plt.show()