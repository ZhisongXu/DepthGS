#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import get_expon_lr_func

# ============================================================
# Local SO(3) <-> axis-angle implementation (NO pytorch3d)
# ============================================================

def matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    R: (..., 3, 3)
    return: (..., 3)  axis * angle
    """
    R = R.float()
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)

    sin_theta = torch.sin(theta).clamp(min=eps)
    axis = axis / (2.0 * sin_theta)[..., None]

    return axis * theta[..., None]


def axis_angle_to_matrix(rot: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rot: (..., 3) axis-angle
    return: (..., 3, 3)
    """
    rot = rot.float()
    angle = torch.linalg.norm(rot, dim=-1, keepdim=True).clamp(min=eps)
    axis = rot / angle

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    ca = torch.cos(angle[..., 0])
    sa = torch.sin(angle[..., 0])
    C = 1.0 - ca

    R = torch.zeros((*rot.shape[:-1], 3, 3), device=rot.device)
    R[..., 0, 0] = ca + x * x * C
    R[..., 0, 1] = x * y * C - z * sa
    R[..., 0, 2] = x * z * C + y * sa

    R[..., 1, 0] = y * x * C + z * sa
    R[..., 1, 1] = ca + y * y * C
    R[..., 1, 2] = y * z * C - x * sa

    R[..., 2, 0] = z * x * C - y * sa
    R[..., 2, 1] = z * y * C + x * sa
    R[..., 2, 2] = ca + z * z * C

    return R


# ============================================================
# Camera
# ============================================================

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, Cx, Cy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 depth=None, R_gt=None, T_gt=None,
                 mat=None, raw_pc=None, kdtree=None):
        super().__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.Cx = Cx
        self.Cy = Cy
        self.image_name = image_name

        self.data_device = torch.device(data_device)

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.depth = depth.to(self.data_device) if depth is not None else None

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.znear = 0.01
        self.zfar = 100.0
        self.trans = trans
        self.scale = scale

        if raw_pc is not None:
            self.raw_pc = raw_pc
        if kdtree is not None:
            self.kdtree = kdtree
        if mat is not None:
            self.mat = mat

        if R_gt is not None and T_gt is not None:
            gt_w2c = torch.tensor(getWorld2View2(R_gt, T_gt, trans, scale)).cuda()
            self.pose_tensor_gt = self.transform_to_tensor(gt_w2c)

        w2c = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        self.pose_tensor = self.transform_to_tensor(w2c)
        self.pose_tensor.requires_grad_(True)

        self.optimizer = torch.optim.Adam(
            [{'params': [self.pose_tensor], 'lr': 0.005, 'name': 'pose'}]
        )

        self.pose_scheduler_args = get_expon_lr_func(
            lr_init=0.02, lr_final=0.002, lr_delay_mult=0.01, max_steps=300
        )

    def update_learning_rate(self, iteration):
        for g in self.optimizer.param_groups:
            if g["name"] == "pose":
                g["lr"] = self.pose_scheduler_args(iteration)
                return g["lr"]

    def get_world_view_transform(self):
        return self.tensor_to_transform(self.pose_tensor).transpose(0, 1)

    def get_full_proj_transform(self):
        w2c = self.get_world_view_transform()
        P = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar,
            fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        return (w2c.unsqueeze(0) @ P.unsqueeze(0)).squeeze(0)

    def get_camera_center(self):
        return self.get_world_view_transform().inverse()[3, :3]

    def get_projection_matrix(self):
        return getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy)

    def transform_to_tensor(self, Tmat, device=None):
        R = Tmat[:3, :3]
        T = Tmat[:3, 3]
        rot = matrix_to_axis_angle(R)
        out = torch.cat([T, rot]).float()
        return out.to(device) if device else out

    def tensor_to_transform(self, tensor):
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        T = tensor[:, :3]
        rot = tensor[:, 3:]
        R = axis_angle_to_matrix(rot)
        RT = torch.cat([R, T[..., None]], dim=2)
        if RT.shape[0] == 1:
            RT = RT[0]
        bottom = torch.tensor([0, 0, 0, 1], device=RT.device).view(1, 4)
        return torch.cat([RT, bottom], dim=0)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(world_view_transform)[3][:3]
