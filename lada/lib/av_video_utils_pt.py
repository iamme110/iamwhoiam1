import av
from typing import Iterator, Tuple
import torch
from av.container.output import OutputContainer
from av.container.input import InputContainer

from lada.lib.video_utils_interface import PytorchVideoReader, PytorchVideoWriter

class PytorchPyavVideoReader(PytorchVideoReader):
    def __init__(self, file, device):
        self.file = file
        self.device = device
        self.container :None | InputContainer = None

    def __enter__(self):
        self.container = av.open(self.file)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def frames(self) -> Iterator[Tuple[torch.Tensor, float]]:
        for frame in self.container.decode(video=0):
            tensor = torch.from_numpy(frame.to_ndarray(format='bgr24'))
            yield tensor, frame.pts

    def seek(self, offset_ns: float):
        offset = int((offset_ns / 1_000_000_000) * av.time_base)
        self.container.seek(offset)


class PytorchPyavVideoWriter(PytorchVideoWriter):
    def get_default_encoder_options(self):
        libx264 = {
            'preset': 'medium',
            'crf': '20'
        }
        libx265 = {
            'preset': 'medium',
            'crf': '23',
            'x265-params': 'log_level=error'
        }
        encoder_defaults = {'libx264': libx264, 'h264': libx264, 'libx265': libx265, 'hevc': libx265}
        return encoder_defaults

    def __init__(self, output_path, width, height, fps, codec, device, crf=None, preset=None, time_base=None, moov_front=False, custom_encoder_options=None):
        # Note: Using av for VideoWriterPT as torchvision.io.write_video is not stream-based
        container_options = {"movflags": "+frag_keyframe+empty_moov+faststart"} if moov_front else {}
        encoder_defaults = self.get_default_encoder_options()
        encoder_options = encoder_defaults.get(codec, {})

        if crf is not None:
            if codec in ('hevc_nvenc', 'h264_nvenc'):
                encoder_options['rc'] = 'constqp'
                encoder_options['qp'] = str(crf)
            else:
                encoder_options['crf'] = str(crf)
        if preset:
            encoder_options['preset'] = preset

        if custom_encoder_options:
            encoder_options.update(self.parse_custom_options(custom_encoder_options))

        output_container: OutputContainer = av.open(output_path, "w", options=container_options)
        video_stream_out: av.VideoStream = output_container.add_stream(codec, fps, options=encoder_options)

        video_stream_out.width = width
        video_stream_out.height = height
        video_stream_out.thread_count = 0
        video_stream_out.thread_type = 3
        video_stream_out.time_base = time_base

        video_stream_out.codec_context.width = width
        video_stream_out.codec_context.height = height
        video_stream_out.codec_context.thread_count = 0
        video_stream_out.codec_context.thread_type = 3
        video_stream_out.codec_context.time_base = time_base

        self.output_container = output_container
        self.video_stream = video_stream_out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def write(self, frame: torch.Tensor, frame_pts=None, bgr2rgb=False, format='rgb24'):
        if bgr2rgb:
            frame = frame[:, :, [2, 1, 0]]
        if frame.dtype != torch.uint8:
            frame = frame.mul_(255.0).byte()
        out_frame = av.VideoFrame.from_ndarray(frame.cpu().numpy(), format=format)
        if frame_pts:
            out_frame.pts = frame_pts
        out_packet = self.video_stream.encode(out_frame)
        self.output_container.mux(out_packet)

    def release(self):
        out_packet = self.video_stream.encode(None)
        self.output_container.mux(out_packet)
        self.output_container.close()
