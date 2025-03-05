import os
import sys
import torch

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from lib.config import (
    DEVICE,
    MODEL_CONFIGS,
    CHECKPOINT_DIR,
    DEPENDENCIES_DIR
)

if DEPENDENCIES_DIR not in sys.path:
    sys.path.insert(1, DEPENDENCIES_DIR)

from dav2.depth_anything_v2.dpt import DepthAnythingV2


def load_model(encoder):
    """Load the Depth Anything V2 model with the given encoder type."""
    print(f"(+) Loading Depth Anything V2 model [{encoder}]...")

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"depth_anything_v2_{encoder}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    print("(+) Model loaded successfully.")

    return model
