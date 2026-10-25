"""Single source of truth for which device model code runs on.

Import ``DEVICE`` everywhere a tensor or module needs to be placed, rather
than checking ``torch.cuda.is_available()`` ad hoc — keeps every call site
consistent and means there's exactly one place to change if this service
is ever pinned to a specific GPU or forced back to CPU.

Falls back to CPU automatically when no GPU is visible to the container
(no NVIDIA Container Toolkit / GPU reservation configured on the host) —
same image, same code, just slower. See train.py's docstring for why CPU
was originally considered acceptable for this tier.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    logger.info("geoops-service: using GPU %s", torch.cuda.get_device_name(DEVICE))
else:
    logger.warning("geoops-service: no CUDA device visible — falling back to CPU")
