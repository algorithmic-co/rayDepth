#!/usr/bin/env python

import aiohttp
import asyncio
import av
import lance
import os
import sys
import ray
import requests
import tarfile
import shutil

import pandas as pd
import pyarrow as pa

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from lib.config import (
    KINETICS_TXT_URLS,
    LOCAL_TMP_DIR,
    LOCAL_TRANSCODED_DIR,
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_DIRECTORY_ID,
    GDRIVE_DIRECTORY_NAME,
    LANCE_DATASET_DIR,
    BATCH_SIZE,
    CHUNK_SIZE,
)
from lib.depth_model_loader import load_model
from lib.depth_utils import extract_depth_maps_from_video
from lib.misc_utils import upload_file_to_gdrive, zip_dir


def fetch_kinetics_url_lists() -> list[str]:
    """
    Fetch the Kinetics .txt files, each containing .tar.gz links.
    Merge them into one big list of URLs.
    """
    all_urls = []
    for url_list in KINETICS_TXT_URLS:
        print(f"Fetching {url_list}")
        r = requests.get(url_list)
        r.raise_for_status()
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        all_urls.extend(lines)
    print(f"Total URLs fetched: {len(all_urls)}")
    return all_urls


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def _async_download(url: str, out_file: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            downloaded = 0
            with open(out_file, "wb") as f:
                while True:
                    chunk = await resp.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)


def download_tar_gz(url: str, out_dir: str) -> str:
    """
    Download .tar.gz from `url` to out_dir, return local path.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(url)
    tar_path = os.path.join(out_dir, filename)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_async_download(url, tar_path))

    return tar_path


def extract_mp4s(tar_path: str, extract_dir: str) -> list[str]:
    """Extract .mp4 files from local tar_path, returning list of local .mp4 paths."""
    os.makedirs(extract_dir, exist_ok=True)
    mp4_paths = []
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".mp4")]
        for m in members:
            tf.extract(m, path=extract_dir)
            mp4_paths.append(os.path.join(extract_dir, m.name))
    return mp4_paths


def transcode_360p(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    container_in = av.open(input_path, mode="r")
    container_out = av.open(output_path, mode="w", format="mp4")

    stream = container_out.add_stream("mpeg4", rate=30)
    for frame in container_in.decode(video=0):
        packet = stream.encode(frame)
        if packet:
            container_out.mux(packet)

    packet = stream.encode(None)
    if packet:
        container_out.mux(packet)

    container_in.close()
    container_out.close()


def estimate_depth(video_path):
    """Run Depth Anything V2 Processing Pipeline..."""
    model = load_model(encoder="vits")
    depth_path = extract_depth_maps_from_video(video_path, model)
    print(f"Successfully stored depth map at: {depth_path}.")
    return depth_path


def process_tar_gz(url: str) -> list[dict]:
    """
    Download -> Extract -> Transcode -> Upload -> Cleanup -> Return metadata
    """
    tar_path = download_tar_gz(url, LOCAL_TMP_DIR)

    base_noext = os.path.splitext(os.path.basename(tar_path))[0]  # e.g. "abseiling"
    extract_dir = os.path.join(LOCAL_TMP_DIR, "extracted", base_noext)
    mp4_paths = extract_mp4s(tar_path, extract_dir)
    os.remove(tar_path)

    results = []
    for mp4_path in mp4_paths:
        base_mp4 = os.path.splitext(os.path.basename(mp4_path))[0]
        out_mp4 = os.path.join(LOCAL_TRANSCODED_DIR, f"{base_mp4}_360p.mp4")
        transcode_360p(mp4_path, out_mp4)

        # remove original
        os.remove(mp4_path)

        # Upload
        # s3_uri = upload_to_s3(out_mp4, S3_BUCKET, S3_PREFIX) - # ideally we would utilize an S3 bucket to upload the data
        mp4_gdrive_url = upload_file_to_gdrive(GDRIVE_CREDENTIALS_PATH,
                                               GDRIVE_DIRECTORY_ID,
                                               GDRIVE_DIRECTORY_NAME,
                                               out_mp4)

        out_depth_path = estimate_depth(out_mp4)
        compressed_depth_path = f"{out_depth_path}.zip"
        zip_dir(out_depth_path, compressed_depth_path)
        depth_gdrive_url = upload_file_to_gdrive(GDRIVE_CREDENTIALS_PATH,
                                                 GDRIVE_DIRECTORY_ID,
                                                 GDRIVE_DIRECTORY_NAME,
                                                 compressed_depth_path)

        os.remove(out_mp4)
        shutil.rmtree(out_depth_path)
        os.remove(compressed_depth_path)

        results.append({
            "original_url": url,
            "transcoded_url": mp4_gdrive_url,
            "depth_url": depth_gdrive_url
        })
    return results


def append_lance(rows: list[dict], lance_dir: str):
    """Append new rows to a Lance dataset. Rows must have consistent keys."""
    if not rows:
        return
    original_url = [r["original_url"] for r in rows]
    transcoded_url = [r["transcoded_url"] for r in rows]
    depth_url = [r["depth_url"] for r in rows]
    table = pa.table({
        "original_url": pa.array(original_url, pa.string()),
        "transcoded_url": pa.array(transcoded_url, pa.string()),
        "depth_url": pa.array(depth_url, pa.string()),
    })
    if not os.path.exists(lance_dir):
        lance.write_dataset(table, lance_dir)
    else:
        lance.write_dataset(table, lance_dir, mode="append")


def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    This is the function passed to map_batches(..., batch_format="pandas").
    batch_df has a single column "url".
    We iterate over each row, call process_tar_gz(url), flatten the results, and
    return a new DataFrame of all metadata records.
    """
    all_results = []
    for url in batch_df["url"]:
        out = process_tar_gz(url)
        all_results.extend(out)
    return pd.DataFrame(all_results)


@ray.remote
def process_chunk(subset):
    ds = ray.data.from_items([{"url": u} for u in subset])
    ds_out = ds.map_batches(
        process_batch,
        batch_size=BATCH_SIZE,
        batch_format="pandas",
    )
    # Gather results back
    rows = ds_out.take_all()
    # Append to Lance
    append_lance(rows, LANCE_DATASET_DIR)
    return rows


def main():
    # Gather all .tar.gz URLs
    all_urls = fetch_kinetics_url_lists()
    if not all_urls:
        print("No URLs found. Exiting.")
        return

    all_urls = all_urls[:4]

    # Start Ray, connect to cluster
    ray.init(address="auto")

    # Chunk the entire URL list to avoid massive concurrency
    chunked = list(chunk_list(all_urls, CHUNK_SIZE))
    print(f"Total chunks: {len(chunked)}. Each chunk has up to {CHUNK_SIZE} URLs")

    # submit tasks in parallel
    chunk_refs = []
    for subset in chunked:
        print(f'Chunk: {subset}')
        chunk_refs.append(process_chunk.remote(subset))

    done, _ = ray.wait(chunk_refs, num_returns=len(chunk_refs), timeout=None)
    if done:
        print("All chunks processed.")

        compressed_lance_path = f"{LANCE_DATASET_DIR}.zip"
        zip_dir(LANCE_DATASET_DIR, compressed_lance_path)
        gdrive_url = upload_file_to_gdrive(GDRIVE_CREDENTIALS_PATH,
                                           GDRIVE_DIRECTORY_ID,
                                           GDRIVE_DIRECTORY_NAME,
                                           compressed_lance_path)
        print(f"Lance dataset uploaded from {compressed_lance_path} to: {gdrive_url}.")

    ray.shutdown()


if __name__ == "__main__":
    main()
