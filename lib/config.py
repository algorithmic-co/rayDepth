#!/usr/bin/env python

import os
import torch


KINETICS_TXT_URLS = [
    "https://s3.amazonaws.com/kinetics/600/val/k600_val_path.txt",
    "https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt",
    "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt",
]

LOCAL_TMP_DIR = "/tmp/kinetics"
LOCAL_TRANSCODED_DIR = "/tmp/kinetics_360p"

LANCE_DATASET_DIR = "/tmp/kinetics_val.lance"

GDRIVE_DIRECTORY_ID = "1g8W9dmApCoJDybYWjbPIUlgC2Sab1Uue"
GDRIVE_DIRECTORY_NAME = "Kinetics"
GDRIVE_CREDENTIALS_PATH = "/app/keys/service_account.json"

CHUNK_SIZE = 1       # number of files per chunk
BATCH_SIZE = 8       # number of parallel workers to limit concurrency


DEPENDENCIES_DIR = "/dependencies"

# Select device dynamically (most likely using CPU due to specs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Depth Anything parameters
CHECKPOINT_DIR = os.path.join(DEPENDENCIES_DIR, "dav2/checkpoints")

# Model configuration
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Processing settings - we've had to add padding since the vision transformer 
# used in depth anything v2 has a 14x14 pixel patch and assert resolution to be multiple of 14.
SKIP_FACTOR = 1
PATCH_MULTIPLE = 14
