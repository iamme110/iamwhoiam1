from typing import TypedDict
from .err import FFmpegError, ConcatSliceError

__all__ = [
    FFmpegError,
    ConcatSliceError
]

class RestoredInfo(TypedDict):
    encoder: str
    encoder_options: str
    slice_file_list: list[str]
    frame_count: int
    ext: str