
import math

import torch
import torch.nn.functional as F


class CachedE2P:
    def __init__(self, height, width, device, fov=(90, 60)):
        x_max = math.tan(math.radians(fov[0]) / 2)
        y_max = math.tan(math.radians(fov[1]) / 2)

        x = torch.linspace(
            -x_max, x_max, width,
            device=device, dtype=torch.float32
        )
        y = torch.linspace(
            y_max, -y_max, height,
            device=device, dtype=torch.float32
        )

        # Reused across all frames
        self.y, self.x = torch.meshgrid(y, x, indexing="ij")

    @torch.inference_mode()
    def __call__(self, frame, yaw, pitch):
        # frame: [C, H, W], float32, CUDA
        h, w = frame.shape[-2:]

        yaw = math.radians(yaw)
        pitch = math.radians(pitch)

        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        # Rotate perspective rays (roll = 0)
        y1 = self.y * cp + sp
        z1 = cp - self.y * sp

        x2 = self.x * cy + z1 * sy
        z2 = z1 * cy - self.x * sy

        # Convert to spherical coordinates
        longitude = torch.atan2(x2, z2)
        latitude = torch.atan2(
            y1, torch.sqrt(x2.square() + z2.square())
        )

        # Match original e2p sampling coordinates
        gx = torch.remainder(
            (longitude / (2 * math.pi) + 0.5) * w - 0.5,
            w
        )
        gy = (
            (-latitude / math.pi + 0.5) * h - 0.5
        ).clamp(0, h - 1)

        # Normalize for align_corners=True
        gx = gx * (2.0 / (w - 1)) - 1.0
        gy = gy * (2.0 / (h - 1)) - 1.0

        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)

        return F.grid_sample(
            frame.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
