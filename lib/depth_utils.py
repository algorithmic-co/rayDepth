import cv2
import torch
import numpy as np
import sys
import os
import pyarrow as pa
import lance
import av
import time

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from lib.config import (
    DEVICE,
    SKIP_FACTOR,
    PATCH_MULTIPLE
)


def resize_to_nearest_multiple(image, multiple=PATCH_MULTIPLE):
    """Resize image dimensions to be a multiple of the given value."""
    h, w = image.shape[:2]
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    if new_h != h or new_w != w:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, (new_h, new_w)


def prepare_image_for_model(image, device):
    """Convert an image to a tensor for model input."""
    resized_image, _ = resize_to_nearest_multiple(image)
    input_tensor = torch.from_numpy(resized_image).float() / 255.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return input_tensor


def extract_depth_maps_from_video(video_path, model, video_id="video_0001"):
    """Extracts depth maps from a video and saves them as images."""
    print(f"(+) Opening video: {video_path}")
    container = av.open(video_path)

    out_dir = video_path.replace('.mp4', '')
    os.makedirs(out_dir, exist_ok=True)

    print("(+) Processing video frames...")
    frame_count = 0

    for frame_idx, frame in enumerate(container.decode(video=0)):
        if frame_idx % SKIP_FACTOR != 0:
            continue

        t0 = time.time()
        rgb_array = frame.to_rgb().to_ndarray()

        input_tensor = prepare_image_for_model(rgb_array, DEVICE)
        
        with torch.no_grad():
            depth_map = model.forward(input_tensor)

        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (rgb_array.shape[1], rgb_array.shape[0]), interpolation=cv2.INTER_LINEAR)

        d_min, d_max = depth_map.min(), depth_map.max()
        if (d_max - d_min) > 1e-8:
            depth_map = (depth_map - d_min) / (d_max - d_min)

        depth_uint8 = (depth_map * 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"depth_frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, depth_uint8)

        frame_count += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"    Processed frame {frame_idx} in {elapsed:.2f}s")

        break

    print(f"(+) Processed {frame_count} frames.")
    return out_dir
