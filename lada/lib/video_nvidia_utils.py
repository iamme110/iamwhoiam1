# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
from typing import Iterator
import PyNvVideoCodec as nvc
from lada.lib.video_utils import VideoReaderBase

class NvidiaVideoReader(VideoReaderBase):
    def __init__(self, file: str, batch_size: int|None, device: torch.device):
        super().__init__(file, batch_size)
        self.decoder_stream = torch.cuda.Stream()
        self.device = device

    def __enter__(self):
        self.decoder = nvc.SimpleDecoder(
            self.file, 
            output_color_type=nvc.OutputColorType.RGBP, 
            gpu_id=self.device.index,
            cuda_stream=self.decoder_stream.cuda_stream,
        )
        self.frames_count = self.decoder.get_stream_metadata().num_frames
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.decoder

    def frames(self) -> Iterator[tuple[torch.Tensor, int]]:
        frame_idx = 0
        
        with torch.cuda.stream(self.decoder_stream):
            while True:
                batch_size = min(self.batch_size, self.frames_count - frame_idx)
                frames = self.decoder.get_batch_frames(batch_size)
                if len(frames) == 0:
                    break
                for f in frames:
                    tensor_nv = torch.from_dlpack(f)
                    tensor_bgr = tensor_nv.permute(1, 2, 0).flip(-1) # (C, H, W) RGB -> (H, W, C) BGR
                    frame_idx += 1
                    yield tensor_bgr, f.getPTS()

    def seek(self, offset_ns: int):
        index = self.decoder.get_index_from_time_in_seconds(offset_ns / 1_000_000_000)
        self.decoder.seek_to_index(index)