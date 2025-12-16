############################################################
# This file is not up-to-date please use examples/train.py
############################################################

import os
import sys
import time
import uuid
import torch
import wandb
import nvidia_smi
import numpy as np
import open3d as o3d

from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from sklearn.neighbors import KDTree

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, from_lowerdiag

from multivariate_normal import CustomMultivariateNormal

# -------------------------------
nvidia_smi.nvmlInit()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

CHUNK_SIZE = 50000

# =========================================================
# Training
# =========================================================
def training(dataset, opt, pipe,
             testing_iterations,
             saving_iterations,
             checkpoint_iterations,
             checkpoint,
             debug_from,
             args):

    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        single_frame_id=args.single_frame_id,
        voxel_size=args.voxel_size,
        init_w_gaussian=args.init_w_gaussian,
        load_ply=args.load_ply
    )

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack = None
    viewpoint_stack_all = scene.getTrainCameras().copy()

    for iteration in range(1, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        depth = render_pkg["depth"]

        gt_image = viewpoint_cam.original_image.to(image.device)
        gt_depth = viewpoint_cam.depth.to(depth.device) if viewpoint_cam.depth is not None else None

        loss = 0.0

        # ---------------- color loss
        Ll1 = l1_loss(image, gt_image)
        color_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss += args.CS * color_loss

        # ---------------- depth loss (ALWAYS ON if DS != None)
        if args.DS is not None and gt_depth is not None:
            gt_depth = gt_depth.clone()
            gt_depth[gt_depth < 0] = 0
            depth = depth.clone()
            depth[0][gt_depth == 0] = 0
            depth_loss = l1_loss(depth, gt_depth)
            loss += args.DS * depth_loss

        loss.backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update(10)

        if iteration in saving_iterations:
            scene.save(iteration)

        if iteration in testing_iterations:
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                0.0,
                testing_iterations,
                scene,
                render,
                (pipe, background)
            )

    progress_bar.close()
    print("\nTraining complete.")

# =========================================================
# Logger
# =========================================================
def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:8])

    os.makedirs(args.model_path, exist_ok=True)

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available")
        return None

# =========================================================
# Eval
# =========================================================
def training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                    elapsed, testing_iterations,
                    scene, renderFunc, renderArgs):

    if iteration not in testing_iterations:
        return

    with torch.no_grad():
        psnr_test = 0.0
        cams = scene.getTestCameras()
        for cam in cams:
            image = renderFunc(cam, scene.gaussians, *renderArgs)["render"]
            gt = cam.original_image.to(image.device)
            psnr_test += psnr(image, gt).mean().item()
        psnr_test /= len(cams)

        print(f"[ITER {iteration}] Test PSNR: {psnr_test:.2f}")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--CS", type=float, default=1.0)
    parser.add_argument("--DS", type=float, default=None)
    parser.add_argument("--load_ply", action="store_true", default=False)
    parser.add_argument("--init_w_gaussian", action="store_true", default=False)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--single_frame_id", type=lambda x: np.array(x.split(",")).astype(int), default=[])
    parser.add_argument("--wandb", action="store_true", default=False)

    # ---- parse
    args = parser.parse_args(sys.argv[1:])

    # ---- SAFETY FIX (核心)
    if not hasattr(args, "iterations"):
        args.iterations = op.iterations

    if not hasattr(args, "save_iterations"):
        args.save_iterations = []

    if not hasattr(args, "test_iterations"):
        args.test_iterations = []

    args.save_iterations.append(args.iterations)

    # ---- setup
    safe_state(False)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        [],
        None,
        -1,
        args
    )

    nvidia_smi.nvmlShutdown()
