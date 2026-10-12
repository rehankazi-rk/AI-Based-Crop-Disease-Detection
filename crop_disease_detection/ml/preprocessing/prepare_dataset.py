from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.download_dataset import extract_and_prepare


if __name__ == "__main__":
    extract_and_prepare()
