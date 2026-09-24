from pathlib import Path

# Καρφωτό, απόλυτο path στο root του project:
PROJ_ROOT = Path(
    "/home/fenia/Desktop/Engineering/Predictive"
    " maintenance/bearing-fault-pred"
)

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = (
    DATA_DIR  / "raw" # ή DATA_DIR / "raw" αν τα xlsx είναι μέσα σε υποφάκελο raw
)
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"