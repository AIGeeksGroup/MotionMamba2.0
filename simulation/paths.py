"""Release-relative paths; no dependency on training result directories."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "simulation"
SOURCE = SIM / "runtime"
LIVE = SIM / "live"
STATS = SIM / "assets" / "stats"
OUTPUT = ROOT / "runtime" / "simulation_previews"


def activate():
    paths = [LIVE, SOURCE / "motion_training/_runtime", SOURCE,
             SIM / "examples", SIM, ROOT / "vendor/state-spaces-mamba"]
    sys.path[:0] = [str(p) for p in paths if str(p) not in sys.path]


def warm_embedding(device):
    import numpy as np
    import torch
    return torch.tensor(np.load(SIM / "assets/warm_text.npy"), device=device)[None, None]
