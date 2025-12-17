############################################################
# Minimal 3DGS training + geometry losses (1~6)
#
# 1) Color loss: L1 + DSSIM
# 2) Depth supervised (if gt depth exists)
# 3) Scale regularization (if gaussians.get_scaling exists)
# 4) Normal consistency (no rendered normals -> depth-normal edge-aware smooth)
# 5) Multi-view color consistency (warp ref render to src, L1 + optional DSSIM)
# 6) Multi-view depth consistency (warp ref depth to src, Charbonnier)
#
# + Densify/Prune/ResetOpacity (best-effort)
#
# IMPORTANT FIX:
# - Avoid argparse conflicts with OptimizationParams(parser) etc.
#   by only adding args if they do NOT already exist.
############################################################

import os
import sys
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import nvidia_smi

from random import randint
from tqdm import tqdm
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state

# -------------------------------
nvidia_smi.nvmlInit()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# =========================================================
# Argparse helper (NO-CONFLICT)
# =========================================================
def add_arg_if_absent(parser: ArgumentParser, *name_or_flags, **kwargs):
    """
    Add argparse option only if it doesn't exist already.
    This prevents: argparse.ArgumentError: conflicting option string
    """
    existing = set()
    for act in parser._actions:
        for opt in getattr(act, "option_strings", []):
            existing.add(opt)

    for flag in name_or_flags:
        if isinstance(flag, str) and flag.startswith("-") and flag in existing:
            return
    parser.add_argument(*name_or_flags, **kwargs)


# =========================================================
# Utils
# =========================================================
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)

def make_K_from_fov(cam, device):
    W, H = cam.image_width, cam.image_height
    fx = W / (2.0 * np.tan(float(cam.FoVx) / 2.0))
    fy = H / (2.0 * np.tan(float(cam.FoVy) / 2.0))
    cx = W * 0.5
    cy = H * 0.5
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], device=device, dtype=torch.float32)
    return K

def get_w2c(cam, device, transpose=False):
    """
    Robust world-to-camera matrix for different GS forks.
    """
    if hasattr(cam, "world_view_transform"):
        w2c = cam.world_view_transform
        if torch.is_tensor(w2c):
            w2c = w2c.to(device).float()
            if transpose:
                w2c = w2c.transpose(0, 1)
            return w2c

    for name in ["w2c", "W2C", "world2cam", "world_view"]:
        if hasattr(cam, name):
            w2c = getattr(cam, name)
            if torch.is_tensor(w2c) and w2c.shape[-2:] == (4, 4):
                w2c = w2c.to(device).float()
                if transpose:
                    w2c = w2c.transpose(0, 1)
                return w2c

    if not (hasattr(cam, "R") and hasattr(cam, "T")):
        raise AttributeError("Camera has no world_view_transform and no R/T")

    R = cam.R
    T = cam.T

    if not torch.is_tensor(R):
        R = torch.tensor(R, device=device, dtype=torch.float32)
    else:
        R = R.to(device).float()

    if not torch.is_tensor(T):
        T = torch.tensor(T, device=device, dtype=torch.float32)
    else:
        T = T.to(device).float()

    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.view(3)

    if transpose:
        w2c = w2c.transpose(0, 1)
    return w2c

def backproject(depth_hw, K, w2c):
    device = depth_hw.device
    H, W = depth_hw.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    x = x.float(); y = y.float()
    z = depth_hw
    valid = (z > 0) & torch.isfinite(z)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (x - cx) / fx * z
    Y = (y - cy) / fy * z

    Pc = torch.stack([X, Y, z, torch.ones_like(z)], dim=-1)  # [H,W,4]
    c2w = torch.linalg.inv(w2c)
    Pw = (Pc.reshape(-1,4) @ c2w.T).reshape(H, W, 4)[..., :3]
    return Pw, valid

def project(Pw, K, w2c, H, W):
    device = Pw.device
    ones = torch.ones((H, W, 1), device=device, dtype=Pw.dtype)
    Pw4 = torch.cat([Pw, ones], dim=-1)  # [H,W,4]
    Pc4 = (Pw4.reshape(-1,4) @ w2c.T).reshape(H, W, 4)

    Xc, Yc, Zc = Pc4[...,0], Pc4[...,1], Pc4[...,2]
    valid = (Zc > 1e-6) & torch.isfinite(Zc)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = fx * (Xc / (Zc + 1e-8)) + cx
    v = fy * (Yc / (Zc + 1e-8)) + cy

    u_norm = 2.0 * (u / (W - 1)) - 1.0
    v_norm = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]
    in_grid = (u_norm.abs() <= 1.0) & (v_norm.abs() <= 1.0)
    return grid, Zc, valid, in_grid

def depth_to_normal(depth_hw, K):
    device = depth_hw.device
    H, W = depth_hw.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device),
                          indexing='ij')
    x = x.float(); y = y.float()
    z = depth_hw

    X = (x - cx) / fx * z
    Y = (y - cy) / fy * z
    P = torch.stack([X, Y, z], dim=0)  # [3,H,W]

    dpx = P[:, :, 2:] - P[:, :, :-2]
    dpy = P[:, 2:, :] - P[:, :-2, :]

    dpx = dpx[:, 1:-1, :]
    dpy = dpy[:, :, 1:-1]

    n = torch.cross(dpx.permute(1,2,0), dpy.permute(1,2,0), dim=-1)  # [H-2,W-2,3]
    n = F.normalize(n, dim=-1, eps=1e-6).permute(2,0,1)
    n = F.pad(n, (1,1,1,1), mode='replicate')
    return n

def edge_weight_from_image(img_chw, k=10.0):
    gx = torch.mean(torch.abs(img_chw[:, :, 1:] - img_chw[:, :, :-1]), dim=0, keepdim=True)
    gy = torch.mean(torch.abs(img_chw[:, 1:, :] - img_chw[:, :-1, :]), dim=0, keepdim=True)
    gx = F.pad(gx, (1,0,0,0), mode='replicate')
    gy = F.pad(gy, (0,0,1,0), mode='replicate')
    g = gx + gy
    w = torch.exp(-k * g).clamp(0, 1)
    return w  # [1,H,W]


# =========================================================
# Logger
# =========================================================
def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output", str(uuid.uuid4())[:8])
    os.makedirs(args.model_path, exist_ok=True)
    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available")
    return None


# =========================================================
# Eval
# =========================================================
def training_report(iteration, testing_iterations, scene, renderFunc, renderArgs):
    if iteration not in testing_iterations:
        return
    with torch.no_grad():
        psnr_test = 0.0
        cams = scene.getTestCameras()
        for cam in cams:
            out = renderFunc(cam, scene.gaussians, *renderArgs)
            img = out["render"]
            gt = cam.original_image.to(img.device)
            psnr_test += psnr(img, gt).mean().item()
        psnr_test /= max(1, len(cams))
        print(f"[ITER {iteration}] Test PSNR: {psnr_test:.2f}")


# =========================================================
# Training
# =========================================================
def training(dataset, opt, pipe, args):
    _ = prepare_output_and_logger(dataset)

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

    viewpoint_stack = None
    viewpoint_stack_all = scene.getTrainCameras().copy()
    cameras_extent = getattr(scene, "cameras_extent", None)

    progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")

    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]  # [3,H,W]
        depth = render_pkg["depth"]
        depth_hw = depth[0] if depth.dim() == 3 else depth

        gt_image = viewpoint_cam.original_image.to(image.device)
        gt_depth = viewpoint_cam.depth.to(depth_hw.device) if getattr(viewpoint_cam, "depth", None) is not None else None

        loss = 0.0

        # 1) color loss
        Ll1 = l1_loss(image, gt_image)
        color_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss + args.CS * color_loss

        # 2) supervised depth loss (if gt exists)
        if args.DS > 0 and gt_depth is not None:
            gt = gt_depth.clone()
            gt[~torch.isfinite(gt)] = 0
            gt[gt < 0] = 0

            d = depth_hw.clone()
            d[~torch.isfinite(d)] = 0
            d[d < 0] = 0

            if args.sup_max_depth is not None:
                gt[gt > args.sup_max_depth] = 0
                d[d > args.sup_max_depth] = 0

            valid = (gt > 0) & (d > 0)
            if valid.any():
                loss = loss + args.DS * charbonnier(d[valid] - gt[valid]).mean()

        # 3) scale regularization
        if args.SC > 0 and hasattr(gaussians, "get_scaling"):
            sc = gaussians.get_scaling
            loss = loss + args.SC * sc.mean()

        # 4) depth-normal edge-aware smooth
        if args.NC > 0:
            device = image.device
            K = make_K_from_fov(viewpoint_cam, device)
            d = depth_hw.clone()
            d[~torch.isfinite(d)] = 0
            d[d < 0] = 0
            if args.nc_max_depth is not None:
                d[d > args.nc_max_depth] = 0

            n = depth_to_normal(d, K)
            w = edge_weight_from_image(gt_image, k=args.NC_edge).detach()

            nx = (n[:, :, 1:] - n[:, :, :-1]).abs().mean(0, keepdim=True)
            ny = (n[:, 1:, :] - n[:, :-1, :]).abs().mean(0, keepdim=True)
            nx = F.pad(nx, (1,0,0,0), mode='replicate')
            ny = F.pad(ny, (0,0,1,0), mode='replicate')
            n_smooth = (w * (nx + ny)).mean()
            loss = loss + args.NC * n_smooth

        # 5/6) multi-view consistency
        if (args.MV_C > 0 or args.MV_D > 0) and len(viewpoint_stack_all) > 1:
            device = image.device
            H, W = depth_hw.shape
            K_src = make_K_from_fov(viewpoint_cam, device)
            w2c_src = get_w2c(viewpoint_cam, device, transpose=args.w2c_transpose)

            d_src = depth_hw.clone()
            d_src[~torch.isfinite(d_src)] = 0
            d_src[d_src < 0] = 0
            if args.mv_max_depth is not None:
                d_src[d_src > args.mv_max_depth] = 0

            Pw, v0 = backproject(d_src, K_src, w2c_src)

            mv_c_acc = 0.0
            mv_d_acc = 0.0
            used = 0

            for _ in range(args.mv_pairs):
                ref_cam = viewpoint_stack_all[randint(0, len(viewpoint_stack_all) - 1)]
                if ref_cam is viewpoint_cam:
                    continue

                ref_pkg = render(ref_cam, gaussians, pipe, background)
                ref_img = ref_pkg["render"].clamp(0, 1)
                ref_dep = ref_pkg["depth"]
                ref_dep = ref_dep[0] if ref_dep.dim() == 3 else ref_dep

                K_ref = make_K_from_fov(ref_cam, device)
                w2c_ref = get_w2c(ref_cam, device, transpose=args.w2c_transpose)

                grid, _, v1, in_grid = project(Pw, K_ref, w2c_ref, H, W)

                ref_img_warp = F.grid_sample(ref_img.unsqueeze(0), grid, align_corners=True).squeeze(0)
                ref_dep_warp = F.grid_sample(ref_dep.unsqueeze(0).unsqueeze(0), grid, align_corners=True).squeeze(0).squeeze(0)

                d_refw = ref_dep_warp.clone()
                d_refw[~torch.isfinite(d_refw)] = 0
                d_refw[d_refw < 0] = 0
                if args.mv_max_depth is not None:
                    d_refw[d_refw > args.mv_max_depth] = 0

                vmask = v0 & v1 & in_grid & (d_src > 0)
                vmask_d = vmask & (d_refw > 0)

                if vmask.sum() < args.mv_min_valid:
                    continue

                if args.MV_C > 0:
                    l1c = charbonnier((image - ref_img_warp)[:, vmask]).mean()
                    if args.mv_c_dssim > 0:
                        if args.mv_c_use_masked_dssim:
                            imgA = image
                            imgB = ref_img_warp.clone()
                            m = vmask.float().unsqueeze(0)
                            imgB = imgB * m + imgA * (1 - m)
                            dss = (1.0 - ssim(imgA, imgB))
                        else:
                            dss = (1.0 - ssim(image, ref_img_warp))
                        mvc = (1.0 - args.mv_c_dssim) * l1c + args.mv_c_dssim * dss
                    else:
                        mvc = l1c
                    mv_c_acc = mv_c_acc + mvc

                if args.MV_D > 0 and vmask_d.sum() >= args.mv_min_valid:
                    mv_d_acc = mv_d_acc + charbonnier(d_src[vmask_d] - d_refw[vmask_d]).mean()

                used += 1

            if used > 0:
                if args.MV_C > 0:
                    loss = loss + args.MV_C * (mv_c_acc / used)
                if args.MV_D > 0:
                    loss = loss + args.MV_D * (mv_d_acc / used)

        # ---- backward & step
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # ======================================================
        # Densification / Pruning / Opacity reset (best-effort)
        # ======================================================
        with torch.no_grad():
            has_stats = (
                hasattr(gaussians, "add_densification_stats") and
                ("viewspace_points" in render_pkg) and
                ("visibility_filter" in render_pkg)
            )
            if has_stats:
                vsp = render_pkg["viewspace_points"]
                vis = render_pkg["visibility_filter"]

                # ---- IMPORTANT FIX: adapt signature
                # Try (vsp, vsp_abs, vis) then fallback to (vsp, vis)
                try:
                    if "viewspace_points_abs" in render_pkg:
                        vsp_abs = render_pkg["viewspace_points_abs"]
                        gaussians.add_densification_stats(vsp, vsp_abs, vis)
                    else:
                        gaussians.add_densification_stats(vsp, vsp, vis)
                except TypeError:
                    gaussians.add_densification_stats(vsp, vis)

                if ("radii" in render_pkg) and hasattr(gaussians, "max_radii2D"):
                    radii = render_pkg["radii"]
                    try:
                        gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                    except Exception:
                        pass

            if (iteration < args.densify_until_iter) and (iteration > args.densify_from_iter) and (iteration % args.densification_interval == 0):
                if hasattr(gaussians, "densify_and_prune"):
                    size_th = args.size_threshold if iteration > args.opacity_reset_interval else None
                    try:
                        gaussians.densify_and_prune(
                            args.densify_grad_threshold,
                            args.densify_abs_grad_threshold,
                            args.opacity_cull_threshold,
                            cameras_extent,
                            size_th
                        )
                    except TypeError:
                        try:
                            gaussians.densify_and_prune(args.densify_grad_threshold)
                        except Exception:
                            pass

            if iteration < args.densify_until_iter and hasattr(gaussians, "reset_opacity"):
                if (iteration % args.opacity_reset_interval == 0) or (dataset.white_background and iteration == args.densify_from_iter):
                    gaussians.reset_opacity()

        # ---- logs
        if iteration % 10 == 0:
            postfix = {"Loss": float(loss.item())}
            if hasattr(gaussians, "get_xyz"):
                postfix["Pts"] = int(gaussians.get_xyz.shape[0])
            progress_bar.set_postfix(postfix)
            progress_bar.update(10)

        if iteration in args.save_iterations:
            scene.save(iteration)

        training_report(iteration, args.test_iterations, scene, render, (pipe, background))

    progress_bar.close()
    print("\nTraining complete.")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # ---- Loss weights
    add_arg_if_absent(parser, "--CS", type=float, default=1.0)
    add_arg_if_absent(parser, "--DS", type=float, default=0.2)
    add_arg_if_absent(parser, "--SC", type=float, default=1e-4)
    add_arg_if_absent(parser, "--NC", type=float, default=0.02)
    add_arg_if_absent(parser, "--MV_C", type=float, default=0.05)
    add_arg_if_absent(parser, "--MV_D", type=float, default=0.2)

    # ---- Depth ranges
    add_arg_if_absent(parser, "--sup_max_depth", type=float, default=None)
    add_arg_if_absent(parser, "--nc_max_depth", type=float, default=10.0)

    # ---- Normal smooth settings
    add_arg_if_absent(parser, "--NC_edge", type=float, default=10.0)

    # ---- MV settings
    add_arg_if_absent(parser, "--mv_pairs", type=int, default=1)
    add_arg_if_absent(parser, "--mv_max_depth", type=float, default=10.0)
    add_arg_if_absent(parser, "--mv_min_valid", type=int, default=2000)
    add_arg_if_absent(parser, "--w2c_transpose", action="store_true", default=False)

    add_arg_if_absent(parser, "--mv_c_dssim", type=float, default=0.0)
    add_arg_if_absent(parser, "--mv_c_use_masked_dssim", action="store_true", default=True)

    # ---- densify/prune knobs
    add_arg_if_absent(parser, "--densify_from_iter", type=int, default=500)
    add_arg_if_absent(parser, "--densify_until_iter", type=int, default=15000)
    add_arg_if_absent(parser, "--densification_interval", type=int, default=100)
    add_arg_if_absent(parser, "--opacity_reset_interval", type=int, default=3000)

    add_arg_if_absent(parser, "--densify_grad_threshold", type=float, default=2e-4)
    add_arg_if_absent(parser, "--densify_abs_grad_threshold", type=float, default=2e-4)
    add_arg_if_absent(parser, "--opacity_cull_threshold", type=float, default=0.005)
    add_arg_if_absent(parser, "--size_threshold", type=float, default=20.0)

    # ---- scene options
    add_arg_if_absent(parser, "--load_ply", action="store_true", default=False)
    add_arg_if_absent(parser, "--init_w_gaussian", action="store_true", default=False)
    add_arg_if_absent(parser, "--voxel_size", type=float, default=None)
    add_arg_if_absent(parser, "--single_frame_id",
                      type=lambda x: np.array(x.split(",")).astype(int),
                      default=[])

    args = parser.parse_args(sys.argv[1:])

    if not hasattr(args, "iterations"):
        args.iterations = op.iterations

    if not hasattr(args, "save_iterations"):
        args.save_iterations = []
    if not hasattr(args, "test_iterations"):
        args.test_iterations = []

    args.save_iterations.append(args.iterations)

    safe_state(False)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args
    )

    nvidia_smi.nvmlShutdown()
