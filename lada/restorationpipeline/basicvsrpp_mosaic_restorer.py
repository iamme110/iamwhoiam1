import logging

import torch

from lada import LOG_LEVEL
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16, clip_length):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.clip_length = clip_length
        self.is_tensorrt_model = model.__class__.__name__ == "GraphModule"

    def restore(self, video: list[torch.Tensor], max_frames=-1) -> list[torch.Tensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        with torch.inference_mode():
            result = []
            inference_view = torch.stack([x.permute(2, 0, 1) for x in video], dim=0).to(device=self.device).to(dtype=self.dtype).div_(255.0).unsqueeze(0)
            if self.is_tensorrt_model and inference_view.shape[1] < self.clip_length:
                pad = self.clip_length - inference_view.shape[1]
                pad_frame = inference_view[:, -1:].repeat(1, pad, 1, 1, 1)
                inference_view = torch.cat([inference_view, pad_frame], dim=1)

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

    def compile(self, output_path: str, max_clip_size: int) -> str:
        import psutil
        import torch_tensorrt

        if max_clip_size > 60:
            logger.warning(f"Max clip size {max_clip_size} is greater than 60. This is not recommended due to increased memory usage and possibly worsened performance for videos with poor mosaic detection.")

        workspace_size = int(psutil.virtual_memory().available * 0.8)
        input = torch.randn(1, max_clip_size, 3, 256, 256, dtype=self.dtype, device=self.device)

        with torch_tensorrt.logging.info():
            logger.info(f"Compiling BasicVSR++ model (TensorRT workspace_size={workspace_size / (1024 ** 3):.2f} GB)")
            trt_gm = torch_tensorrt.compile(
                self.model, 
                ir="dynamo", 
                inputs=[input],
                min_block_size=1,
                workspace_size=workspace_size,
                enabled_precisions={self.dtype},
                use_fp32_acc=False,
                use_explicit_typing=False,
                sparse_weights=False,
                optimization_level=3,
                hardware_compatible=False,
                use_python_runtime=False,
                cache_built_engines=False,
                reuse_cached_engines=False,
                truncate_double=True)

        torch_tensorrt.save(trt_gm, output_path, inputs=[input])
        del trt_gm
        del input
        return output_path
