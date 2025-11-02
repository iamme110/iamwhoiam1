import logging

import numpy as np
import torch
from lada.basicvsrpp.mmagic.registry import MODELS
from lada.basicvsrpp import register_all_modules
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from lada.lib import Image, ImagePt
from lada.lib.image_utils import img2tensor, tensor2img

logger = logging.getLogger(__name__)

def get_default_gan_inference_config() -> dict:
    return dict(
        type='BasicVSRPlusPlusGan',
        generator=dict(
            type='BasicVSRPlusPlusGanNet',
            mid_channels=64,
            num_blocks=15,
            spynet_pretrained=None),
        pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
        is_use_ema=True,
        data_preprocessor=dict(
            type='DataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
        ))


def load_model(config: str | dict | None, checkpoint_path, device):
    register_all_modules()
    if device and type(device) == str:
        device = torch.device(device)
    if type(config) == str:
        config = Config.fromfile(config).model
    elif type(config) == dict:
        pass
    else:
        raise Exception("unsupported value for 'config', Must be either a file path to a config file or a dict definition of the model")
    model = MODELS.build(config)
    load_checkpoint(model, checkpoint_path, map_location='cpu', logger=logger)
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference(model, video: list, device, max_frames=-1):
    input_frame_count = len(video)
    input_frame_shape = video[0].shape
    if device and type(device) == str:
        device = torch.device(device)
    with torch.no_grad():
        result = []
        input = torch.stack(img2tensor(video, bgr2rgb=False, float32=True), dim=0)
        input = torch.unsqueeze(input, dim=0)  # TCHW -> BTCHW
        if max_frames > 0:
            for i in range(0, input.shape[1], max_frames):
                output = model(inputs=input[:, i:i + max_frames].to(device))
                result.append(output)
            result = torch.cat(result, dim=1)
        else:
            result = model(inputs=input.to(device))
        result = torch.squeeze(result, dim=0)  # BTCHW -> TCHW
        result = list(torch.unbind(result, 0))
        output = tensor2img(result, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))
        output_frame_count = len(output)
        output_frame_shape = output[0].shape
        assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape
        return output

def inference_pt(model, video: list[ImagePt], device, max_frames=-1):
    if not video:
        raise ValueError("Video list cannot be empty")

    with torch.no_grad():
        # Prepare input tensor: stack, normalize, permute to TCHW, add batch dim to BTCHW
        input_tensor = torch.stack([frame for frame in video], dim=0).to(device=torch.device(device), dtype=torch.float32).div_(255.0).permute(0, 3, 1, 2).unsqueeze(0)

        if max_frames > 0:
            outputs = []
            for i in range(0, input_tensor.shape[1], max_frames):
                chunk = input_tensor[:, i:i + max_frames]
                output_chunk = model(inputs=chunk)
                outputs.append(output_chunk)
            result = torch.cat(outputs, dim=1)
        else:
            result = model(inputs=input_tensor)

        return [frame.permute(1, 2, 0) for frame in result.squeeze(0).unbind(0)]

def test():
    device = "cuda:0"

    model = load_model("configs/basicvsrpp/mosaic_restoration_generic_stage2.py",
                       "experiments/basicvsrpp/mosaic_restoration_generic_stage2/iter_100000.pth", device)

    frame1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame3 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    frame4 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    video = [frame1, frame2, frame3, frame4]
    result = inference(model, video, device)
    print(len(result), result[0].shape)


if __name__ == '__main__':
    test()
