import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

    @staticmethod
    def _get_temporal_inference_length(frame_count: int, max_frame_count: int,
                                       temporal_bucket_size: int) -> int:
        if temporal_bucket_size <= 0 or frame_count <= 8:
            return frame_count
        bucket_length = ((frame_count + temporal_bucket_size - 1) // temporal_bucket_size) * temporal_bucket_size
        return min(bucket_length, max_frame_count)

    @staticmethod
    def _pad_video_temporally(video: list[ImageTensor], target_frame_count: int) -> list[ImageTensor]:
        """Reflect frames at the end so MPS sees only a bounded set of temporal shapes."""
        if target_frame_count <= len(video):
            return video
        if len(video) == 1:
            return video * target_frame_count

        result = list(video)
        reflection_period = 2 * (len(video) - 1)
        for position in range(len(video), target_frame_count):
            reflected_position = position % reflection_period
            frame_index = (
                reflected_position
                if reflected_position < len(video)
                else reflection_period - reflected_position
            )
            result.append(video[frame_index])
        return result

    def restore(self, video: list[ImageTensor], max_frames=-1, temporal_bucket_size=0,
                max_frame_count=None) -> list[ImageTensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        if max_frame_count is None:
            max_frame_count = input_frame_count
        inference_frame_count = self._get_temporal_inference_length(
            input_frame_count, max_frame_count, temporal_bucket_size
        )
        inference_video = self._pad_video_temporally(video, inference_frame_count)
        with torch.inference_mode():
            result = []
            inference_view = torch.stack([x.permute(2, 0, 1) for x in inference_video], dim=0).to(device=self.device).to(dtype=self.dtype).div_(255.0).unsqueeze(0)

            if max_frames > 0:
                for i in range(0, inference_view.shape[1], max_frames):
                    output = self.model(inputs=inference_view[:, i:i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self.model(inputs=inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            # Keep every MPS operation on the bucketed temporal shape. Slicing on MPS here
            # would compile a new graph for every original clip length and defeat bucketing.
            result = result.squeeze(0) # -> (bucketed T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (bucketed T, H, W, C)
            result = list(torch.unbind(result, 0))[:input_frame_count] # fixed-shape MPS unbind, variable-length Python list
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result
