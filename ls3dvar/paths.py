"""
ls3dvar/paths.py

Repository locations. The package lives next to mantari/ and src/:

    <repo>/ls3dvar/      this package
    <repo>/src/          HRRRCast source (resnet, fcst, diffusion_params)
    <repo>/net-diffusion/model.keras
    <repo>/mantari/config/test.yaml   default config (only the YAML file is read; no mantari code)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__import__("os").environ.get("HRRRCAST_REPO") or next((p for p in Path(__file__).resolve().parents if (p / "src" / "resnet.py").exists()), Path(__file__).resolve().parents[1]))
SRC_DIR = REPO_ROOT / "src"
MODEL_PATH = REPO_ROOT / "net-diffusion" / "model.keras"
DEFAULT_CONFIG = REPO_ROOT / "mantari" / "config" / "test.yaml"


def add_hrrrcast_src():
    """make `import resnet`, `import fcst` work (same as the old sys.path.insert lines)"""
    for p in (str(REPO_ROOT), str(SRC_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
