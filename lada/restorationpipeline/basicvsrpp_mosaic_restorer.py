import logging

import torch

from lada import LOG_LEVEL
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool, clip_length: int):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.clip_length = clip_length
        self.is_tensorrt_model = model.__class__.__name__ == "GraphModule"

    def restore(self, video: list[ImageTensor], max_frames=-1) -> list[ImageTensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        with torch.inference_mode():
            result = []
            inference_view = torch.stack([x.permute(2, 0, 1) for x in video], dim=0).to(device=self.device).to(dtype=self.dtype).div_(255.0).unsqueeze(0)
            if self.is_tensorrt_model and inference_view.shape[1] < self.clip_length:
                t = inference_view.shape[1]
                if t == 1:
                    idx = torch.zeros(self.clip_length, dtype=torch.long, device=inference_view.device)
                else:
                    base = list(range(t)) + list(range(t - 2, 0, -1))
                    reps = (self.clip_length + len(base) - 1) // len(base)
                    idx = torch.tensor((base * reps)[:self.clip_length], dtype=torch.long, device=inference_view.device)
                inference_view = inference_view.index_select(1, idx)

            if max_frames > 0:
                for i in range(0, inference_view.shape[1], max_frames):
                    output = self.model(inputs=inference_view[:, i:i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self.model(inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            result = result.squeeze(0)[:input_frame_count] # -> (T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
            result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result
