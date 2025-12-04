import numpy as np
import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import Image, ImageTensor
from lada.utils.image_utils import img2tensor, tensor2img

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16, clip_length):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

    def restore(self, video: list[Image | ImageTensor]) -> list[Image | ImageTensor]:
        input_frame_count = len(video)
        if input_frame_count == 0:
            return []
        input_frame_shape = video[0].shape
        is_image_tensor = isinstance(video[0], torch.Tensor)
        with torch.inference_mode():
            if is_image_tensor:
                input = torch.stack([x.permute(2, 0, 1) for x in video]).to(device=self.device, dtype=self.dtype)
                input.div_(255.0)
            else:
                input = torch.stack(img2tensor(video, bgr2rgb=False, float32=True), dim=0).to(device=self.device, dtype=self.dtype)
            input.unsqueeze_(0)  # TCHW -> BTCHW

            result = self.model(inputs=input)

            result.squeeze_(0) # BTCHW -> TCHW
            if is_image_tensor:
                # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
                result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
                result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
            else:
                result = list(torch.unbind(result, 0))
                result = tensor2img(result, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))

            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result
