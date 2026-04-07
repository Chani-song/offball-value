from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/metrica-sports/sample-data.git"
TARGET = Path("data/raw/metrica-sample-data")


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"[skip] {TARGET} already exists")
    else:
        subprocess.run([
            "git",
            "clone",
            "--depth",
            "1",
            REPO_URL,
            str(TARGET),
        ], check=True)
        print(f"[ok] cloned into {TARGET}")
