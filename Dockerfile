# Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    ffmpeg \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavcodec-extra \
    libx264-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    "ray[default]" "ray[data]" \
    pyav \
    aiohttp \
    requests \
    boto3 \
    s3fs \
    pydrive \
    pylance \
    pydrive2 \
    tqdm \
    torch

RUN pip install --no-cache-dir \
    matplotlib \
    opencv-python \
    open3d \
    torchvision

WORKDIR /app
COPY . /app

RUN mv /app/dependencies /dependencies

CMD [ "python", "/app/src/run_pipeline.py" ]
