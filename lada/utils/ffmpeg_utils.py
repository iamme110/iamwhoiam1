from collections import deque
import heapq
import shlex
import subprocess
import sys
from typing import Literal

import cv2
import numpy as np
import torch


class FFVideoWriter:
    def _parse_encoder_options(self, encoder_options: str):
        tokens = shlex.split(encoder_options)
        parsed_encoder_options = [
            (tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)
        ]

        opts = []
        for kv in parsed_encoder_options:
            opts.extend(kv)

        return opts

    def _parse_input_options(self):
        match self.format:
            case "mjpeg":
                return [
                    "-f",
                    "mjpeg",
                    "-r",
                    f"{self.fps:.2f}",
                ]
            case "rawvideo":
                return [
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",  # use bgr instead of rgb to avoid cvtColor cost
                    "-s",
                    f"{self.width}x{self.height}",
                    "-r",
                    f"{self.fps:.2f}",
                ]
            case _:
                raise ValueError(f"unsupported image format {self.format}")

    def _parse_filter_options(self):
        options = []
        if self.time_base is not None:
            options.append(f"settb={self.time_base}")

        return ["-vf", ",".join(options)] if options else []

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        encoder: str,
        encoder_options: str,
        time_base: str | None = None,
        mp4_fast_start: bool = False,
        format: str = "mjpeg",
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format
        self.time_base = time_base

        input_opts = self._parse_input_options()
        output_opts = self._parse_encoder_options(encoder_options)
        if mp4_fast_start and output_path.lower().endswith((".mp4", ".mov")):
            output_opts.extend(("-movflags", "+frag_keyframe+empty_moov+faststart"))
        filter_opts = self._parse_filter_options()

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            *input_opts,
            "-i",
            "-",
            *filter_opts,
            *output_opts,
            output_path,
        ]
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10**7,
        )

        assert proc.stdin is not None
        self.proc = proc
        self.writer = proc.stdin

        # Buffers for reordering frames
        self.BUFFER_MAX_SIZE = 30
        self.pts_heap = []
        self.frame_queue = deque[np.ndarray]()
        self.pts_set = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def _encode_frame(self, frame: np.ndarray):
        match self.format:
            case "mjpeg":
                ok, frame = cv2.imencode(".jpg", frame)
            case _:
                ok = True
        assert ok, "frame encoding failed"

        return frame

    def _process_buffer(self, flush_all=False):
        """Processes the buffer to encode frames."""
        if len(self.frame_queue) > (self.BUFFER_MAX_SIZE / 2) or (
            flush_all and self.frame_queue
        ):
            frame_to_encode = self.frame_queue.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)

            out_frame = frame_to_encode.tobytes()
            self.writer.write(out_frame)

    def write(self, frame, frame_pts=None, is_bgr=False):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if not is_bgr:
            # here we inverse original bgr to rgb pipeline, since ffmpeg supports bgr inputs
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = self._encode_frame(frame)

        if frame_pts and frame_pts not in self.pts_set:
            heapq.heappush(self.pts_heap, frame_pts)
            self.frame_queue.append(frame)
            self.pts_set.add(frame_pts)

        self._process_buffer()

    def release(self):
        while len(self.frame_queue) > 0:
            self._process_buffer(flush_all=True)
        self.writer.close()
        self.proc.wait()
