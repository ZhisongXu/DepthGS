#
# Copyright (C) 2023, Inria
# GRAPHDECO research group
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, GaussianModel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import numpy as np

import cv2
import matplotlib.pyplot as plt

TUM = 0


def list_of_ints(arg):
    return np.array(arg.split(',')).astype(int)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path    = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    depth_path  = os.path.join(model_path, name, f"ours_{iteration}", "depth")
    alpha_path  = os.path.join(model_path, name, f"ours_{iteration}", "alpha")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(alpha_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        results = render(view, gaussians, pipeline, background)

        rendering = results["render"]
        depth     = results["depth"]
        alpha     = results["alpha"]

        gt = view.original_image[0:3]

        max_depth = 50.0
        depth = depth / max_depth

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt,        os.path.join(gts_path,    f"{idx:05d}.png"))
        torchvision.utils.save_image(depth,     os.path.join(depth_path,  f"{idx:05d}.png"))
        torchvision.utils.save_image(alpha,     os.path.join(alpha_path,  f"{idx:05d}.png"))


def render_mask_set(model_path, iteration, test_views, train_views, gaussians, pipeline, background):
    base = os.path.join(model_path, "test_masked", f"ours_{iteration}")
    render_path = os.path.join(base, "renders")
    gt_path     = os.path.join(base, "gt")
    mask_path   = os.path.join(base, "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    print("[Render] Building seen-mask from train point clouds")

    pc_all = []
    for v in train_views:
        if hasattr(v, "raw_pc") and v.raw_pc is not None:
            pc_all.append(torch.tensor(v.raw_pc))

    if len(pc_all) == 0:
        print("[Render] No raw_pc found, skip mask rendering")
        return

    pc = torch.cat(pc_all, dim=0).float().cuda()

    for idx, view in enumerate(tqdm(test_views, desc="Rendering masked test")):
        camera_pose = torch.tensor(view.mat).float().cuda()
        projmatrix  = view.get_projection_matrix().float().cuda()

        W, H = view.image_width, view.image_height

        xyz_h = torch.cat([pc, torch.ones(len(pc), 1, device="cuda")], dim=1)
        p = (xyz_h @ (projmatrix @ torch.inverse(camera_pose)).T)
        p = p / p[:, 3:4]

        uv = p[:, :2]
        uv[:, 0] = (uv[:, 0] + 1) * W * 0.5
        uv[:, 1] = (uv[:, 1] + 1) * H * 0.5
        uv = uv.long()

        mask = torch.zeros((H, W), device="cuda")
        valid = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H)
        uv = uv[valid]
        mask[uv[:, 1], uv[:, 0]] = 1.0

        mask = mask.cpu().numpy()
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, 2)
        mask = cv2.erode(mask, kernel, 2)
        mask = torch.tensor(mask, device="cuda").bool()

        results = render(view, gaussians, pipeline, background)

        rendering = results["render"] * mask
        gt        = view.original_image[0:3] * mask

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt,        os.path.join(gt_path,     f"{idx:05d}.png"))
        torchvision.utils.save_image(mask.float(), os.path.join(mask_path, f"{idx:05d}.png"))


def render_sets(dataset, iteration, pipeline, single_frame_id, use_pseudo_cam):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            single_frame_id=single_frame_id,
            load_ply=True,
            use_pseudo_cam=use_pseudo_cam
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_views = scene.getTrainCameras()
        test_views  = scene.getTestCameras()

        print(f"[Render] Train cameras: {len(train_views)}")
        print(f"[Render] Test cameras : {len(test_views)}")

        if len(train_views) > 0:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       train_views, gaussians, pipeline, background)

        if len(test_views) > 0:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       test_views, gaussians, pipeline, background)

            render_mask_set(dataset.model_path, scene.loaded_iter,
                            test_views, train_views,
                            gaussians, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render all views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--single_frame_id", type=list_of_ints, default=[])
    parser.add_argument("--use_pseudo_cam", action="store_true")
    parser.add_argument("--TUM", action="store_true")

    args = get_combined_args(parser)
    TUM = args.TUM

    print("Rendering:", args.model_path)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.single_frame_id,
        args.use_pseudo_cam
    )
