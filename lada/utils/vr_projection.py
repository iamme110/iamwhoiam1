import torch
import torch.nn.functional as F
import numpy as np


class FisheyeToVR180:
    def __init__(self, height: int, width: int):
        """
        Converts Dual Fisheye images back to a single VR180 SBS (Equirectangular) frame.

        Args:
            height: Height of the full OUTPUT SBS frame.
            width: Width of the full OUTPUT SBS frame.
        """
        self.h = height
        self.w = width
        self.eye_w = width // 2
        self.eye_h = height
        self.grid = None

    def _generate_grid(self, H, W):
        # Output Coordinates (Equirectangular) [-1, 1]
        y_rng = torch.linspace(-1, 1, H)
        x_rng = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(y_rng, x_rng, indexing='ij')

        # Convert Equirect UV to Longitude/Latitude
        lon = grid_x * (np.pi / 2)
        lat = grid_y * (np.pi / 2)

        # Convert Long/Lat to 3D Vector
        x = torch.cos(lat) * torch.sin(lon)
        y = torch.sin(lat)
        z = torch.cos(lat) * torch.cos(lon)

        # Convert 3D Vector to Fisheye Polar Coords
        # Optical axis is Z-axis
        theta = torch.acos(z.clamp(-1, 1))
        r = theta / (np.pi / 2)
        phi_img = torch.atan2(y, x)

        # Convert Polar to Source (Fisheye) Texture UV [-1, 1]
        u = r * torch.cos(phi_img)
        v = r * torch.sin(phi_img)

        # Stack -> (1, H, W, 2)
        grid = torch.stack((u, v), dim=-1).unsqueeze(0)

        # Expand to Batch=2 (Left eye + Right eye)
        return grid.expand(2, -1, -1, -1)

    def __call__(self, left_eye: torch.Tensor, right_eye: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left_eye: Tensor (H, W/2, C) - Left Fisheye Image
            right_eye: Tensor (H, W/2, C) - Right Fisheye Image

        Returns:
            Tensor (H, W, C): Recombined Side-by-Side VR180 frame.
        """
        if self.grid is None:
            self.grid = self._generate_grid(self.eye_h, self.eye_w).to(left_eye)

        x = torch.stack((left_eye, right_eye), dim=0)

        # Permute to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        if not x.is_contiguous():
            x = x.contiguous()

        out = F.grid_sample(x, self.grid, align_corners=True, padding_mode='zeros')

        # Permute back to (N, H, W, C)
        out = out.permute(0, 2, 3, 1)

        # Reconstruct Side-by-Side (H, W_total, C)
        # Move Batch dim next to Width: (2, H, W, C) -> (H, 2, W, C)
        out = out.transpose(0, 1)

        # Flatten (2, W) into (2*W): (H, 2*W, C)
        out = out.reshape(self.h, self.w, -1)

        return out


class VR180ToFisheye:
    def __init__(self, height: int, width: int):
        """
        Converts VR180 SBS (Equirectangular) video to Dual Fisheye (Equidistant).

        Args:
            height: Height of the full SBS frame.
            width: Width of the full SBS frame.
        """
        self.h = height
        self.w = width
        self.eye_w = width // 2
        self.eye_h = height

        # Pre-calculate grid if device is known
        self.grid = None

    def _generate_grid(self, H, W):
        # Create coordinates for the Target (Fisheye) Image [-1, 1]
        y_rng = torch.linspace(-1, 1, H)
        x_rng = torch.linspace(-1, 1, W)
        grid_y, grid_x = torch.meshgrid(y_rng, x_rng, indexing='ij')

        # Polar conversion
        r = torch.sqrt(grid_x.pow(2) + grid_y.pow(2))
        phi = torch.atan2(grid_y, grid_x)

        # Equidistant mapping: theta = r * (pi/2)
        theta = r * (np.pi / 2)

        # Spherical vectors
        sin_theta = torch.sin(theta)
        vec_x = sin_theta * torch.cos(phi)
        vec_y = sin_theta * torch.sin(phi)
        vec_z = torch.cos(theta)

        # 5. Equirectangular UVs
        src_lambda = torch.atan2(vec_x, vec_z)
        src_varphi = torch.asin(vec_y)

        # Normalize
        u = src_lambda / (np.pi / 2)
        v = src_varphi / (np.pi / 2)

        # Mask out-of-bound pixels (outside fisheye circle)
        mask = r > 1.0
        u[mask] = 2.0
        v[mask] = 2.0

        # Stack -> (1, H, W, 2)
        grid = torch.stack((u, v), dim=-1).unsqueeze(0)

        return grid.expand(2, -1, -1, -1)

    def __call__(self, frame_sbs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_sbs: Tensor (H, W, C) - Full SBS Equirectangular frame. Assumes float32.
        Returns:
            Tuple[Tensor, Tensor]: (left_eye, right_eye) (H, W/2, C)
        """
        if self.grid is None:
            self.grid = self._generate_grid(self.eye_h, self.eye_w).to(frame_sbs.device)

        H, W_total, C = frame_sbs.shape

        # Reshape to (H, 2, W_eye, C) effectively splitting eyes without copy
        x = frame_sbs.view(H, 2, self.eye_w, C)

        # Permute to (Batch=2, Channel, Height, Width)
        x = x.permute(1, 3, 0, 2)

        if not x.is_contiguous():
            x = x.contiguous()

        out = F.grid_sample(x, self.grid, align_corners=True, padding_mode='zeros')

        # (N, C, H, W) -> (N, H, W, C)
        out = out.permute(0, 2, 3, 1)

        return out[0], out[1]