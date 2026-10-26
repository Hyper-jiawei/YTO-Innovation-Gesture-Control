"""Use project-local CUDA packages before user-site CPU packages."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / 'outputs' / 'yolo_config').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs' / 'yolo_config'))
for folder in ('.extras', '.deps'):
    path = ROOT / folder
    if path.exists():
        sys.path.insert(0, str(path))
