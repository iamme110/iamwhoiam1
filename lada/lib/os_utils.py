# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import subprocess
import sys
import torch

def get_subprocess_startup_info():
    if sys.platform != "win32":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info

def is_valid_nvidia_gpu_available() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        import PyNvVideoCodec as nvc
    except ImportError:
        return False

    min_vram_gb = 7.5
    cc_major, _ = torch.cuda.get_device_capability()
    total_vram = torch.cuda.get_device_properties().total_memory
    vram_gb = total_vram / (1024 ** 3)
    is_maxwell_or_newer = (cc_major >= 5)
    has_required_vram = (vram_gb >= min_vram_gb)

    return is_maxwell_or_newer and has_required_vram