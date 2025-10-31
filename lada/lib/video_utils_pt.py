import abc
import torch
from torchvision.transforms import functional as torch_functional
from fractions import Fraction

from lada.lib import ImagePt, MaskPt

class VideoReaderPT(abc.ABC):
    @abc.abstractmethod
    def __init__(self, file):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def frames(self):
        pass

    @abc.abstractmethod
    def seek(self, offset_sec):
        pass

class VideoWriterPT(abc.ABC):
    @abc.abstractmethod
    def __init__(self, file, width, height, fps, codec, crf=None, preset=None, time_base=None, moov_front=False, custom_encoder_options=None):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def write(self, frame, frame_pts=None, bgr2rgb=False):
        pass

    @abc.abstractmethod
    def release(self):
        pass

    @abc.abstractmethod
    def get_default_encoder_options(self):
        pass

    @staticmethod
    def parse_custom_options(custom_encoder_options):
        # squeeze spaces
        custom_encoder_options = ' '.join(custom_encoder_options.split())
        import re
        regex = re.compile(r"-(\w+ \w+)")
        matches = regex.findall(custom_encoder_options)
        encoder_options = {}
        for match in matches:
            option, value = match.split()
            encoder_options[option] = value
        return encoder_options

from .av_video_utils_pt import AVVideoReaderPT, AVVideoWriterPT

NowVideoReaderPT = AVVideoReaderPT
NowVideoWriterPT = AVVideoWriterPT

def read_video_frames_pt(path: str, float32: bool = True, start_idx: int = 0, end_idx: int | None = None, normalize_neg1_pos1=False, binary_frames=False) -> list[ImagePt]:
    with NowVideoReaderPT(path) as video_reader:
        frames = []
        i = 0
        for frame, pts in video_reader.frames():
            if i >= start_idx and (end_idx is None or i < end_idx):
                if binary_frames:
                    frame = torch_functional.rgb_to_grayscale(frame)
                    frame = frame.unsqueeze(-1)
                if float32:
                    frame = frame.float() / 255.0
                    if normalize_neg1_pos1:
                        frame = (frame - 0.5) / 0.5
                frames.append(frame)
            i += 1
            if end_idx is not None and i >= end_idx:
                break
    return frames

def resize_video_frames_pt(frames: list[ImagePt], size: int | tuple[int, int]) -> list[ImagePt]:
    resized = []
    if isinstance(size, int):
        target_size = [size, size]
    else:
        target_size = list(size)
    for frame in frames:
        if frame.shape[:2] == target_size:
            resized.append(frame)
        else:
            # Permute to (C, H, W) for F.resize
            frame_chw = frame.permute(2, 0, 1)
            resized_chw = torch_functional.resize(frame_chw, target_size, interpolation=torch_functional.InterpolationMode.BILINEAR)
            resized_frame = resized_chw.permute(1, 2, 0)
            resized.append(resized_frame)
    return resized


def pad_to_compatible_size_for_video_codecs_pt(imgs: list[ImagePt]) -> list[ImagePt]:
    # dims need to be divisible by 2 by most codecs. given the chroma / pix format dims must be divisible by 4
    h, w = imgs[0].shape[:2]
    pad_h = 0 if h % 4 == 0 else 4 - (h % 4)
    pad_w = 0 if w % 4 == 0 else 4 - (w % 4)
    if pad_h == 0 and pad_w == 0:
        return imgs
    else:
        padded = []
        for img in imgs:
            # Pad: (pad_C_left, pad_C_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right)
            padded_img = torch.nn.functional.pad(img, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
            padded.append(padded_img)
        return padded


def write_frames_to_video_file_pt(frames: list[ImagePt], output_path, fps: int | float | Fraction, codec='libx264', preset='medium', crf=None):
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    with NowVideoWriterPT(output_path, width, height, fps, codec, crf=crf, preset=preset) as writer:
        for frame in frames:
            writer.write(frame)


def write_masks_to_video_file_pt(frames: list[MaskPt], output_path, fps: int | float | Fraction):
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    with NowVideoWriterPT(output_path, width, height, fps, 'ffv1', custom_encoder_options='-level 3') as writer:
        for frame in frames:
            if frame.dim() == 3 and frame.shape[2] == 1:
                frame = frame.squeeze(2)
            if frame.dtype != torch.uint8:
                frame = (frame * 255).byte()
            writer.write(frame, format='gray')
