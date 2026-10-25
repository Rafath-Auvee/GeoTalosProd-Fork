# geoops-service — GPU model-training/serving microservice.
#
# Plain Python base is enough even for GPU use: the cu124 torch/torchvision
# wheels (requirements.txt) bundle their own CUDA runtime libraries, and the
# NVIDIA Container Toolkit mounts the host driver's libcuda.so etc. into the
# container at run time (via the `nvidia` runtime + GPU reservation in
# docker-compose.yml) — no CUDA base image (e.g. nvidia/cuda or
# pytorch/pytorch) required.
FROM python:3.11-slim

# libgl1/libglib2.0-0 are opencv-python's runtime shared-lib deps (pulled in
# transitively by ultralytics, used for SAM3 inference) — opencv-python's
# wheel doesn't bundle these, and without them `import cv2` fails with
# "libxcb.so.1: cannot open shared object file" on this slim base.
# git is needed at build time to pip-install requirements.txt's CLIP
# dependency from its GitHub URL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ONNX export of trained heads (app/onnx_export.py). A separate layer so the heavy torch layer stays cached.
RUN pip install --no-cache-dir onnx==1.17.0

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
