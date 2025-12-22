# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import subprocess
import sys

import torch

import logging
from lada import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def get_subprocess_startup_info():
    if sys.platform != "win32":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info

def has_modern_nvidia_gpu(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 7:
        # No tensor cores
        return False
    if major > 7:
        return True
    name = torch.cuda.get_device_name(device_index).lower()
    if "gtx 16" in name:
        return False
    return True

def supports_tensorrt() -> bool:
    if not has_modern_nvidia_gpu():
        return False

    try:
        logging.getLogger("torch_tensorrt").setLevel(logging.ERROR) 
        import torch_tensorrt
        return True
    except ImportError:
        return False

def get_gpu_vram_gb(device: str | torch.device) -> float:
    device = torch.device(device) if isinstance(device, str) else device
    assert device.type == "cuda"
    device_index = 0 if device.index is None else device.index
    return torch.cuda.get_device_properties(device_index).total_memory / (1024**3)