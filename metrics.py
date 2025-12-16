#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import sse, psnr, masked_psnr
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readMasks(masks_dir):
    masks = []
    image_names = []
    for fname in sorted(os.listdir(masks_dir)):
        mask = Image.open(masks_dir / fname)
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return masks, image_names

def evaluate(model_paths, mask_type, split="test", viz=False):

    img_eval_dict = {}
    full_dict = {}
    per_view_dict = {}

    for scene_dir in model_paths:
        print("Scene:", scene_dir)

        img_eval_dict[scene_dir] = {}
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        suffix = ""

        # ✅ split directory
        # - no_mask: <scene>/<split>
        # - mask:    <scene>/<split>_masked
        # - seen_mask: test_seen_masked  (一般只对 test 有意义，但你要 train 也行)
        if mask_type == "seen_mask":
            test_dir = Path(scene_dir) / f"{split}_seen_masked"
            suffix = f"_{split}_seen_masked"
        elif mask_type == "eroded_seen_mask":
            test_dir = Path(scene_dir) / "eroded_seen_mask" / f"{split}_seen_masked"
            suffix = f"_{split}_eroded_seen_masked"
        elif mask_type == "mask":
            test_dir = Path(scene_dir) / f"{split}_masked"
            suffix = f"_{split}_masked"
        else:
            test_dir = Path(scene_dir) / split
            suffix = f"_{split}"

        if not test_dir.exists():
            raise FileNotFoundError(f"[metrics] split dir not found: {test_dir}")

        for method in os.listdir(test_dir):
            if method != "ours_" + str(args.iterations):
                continue
            print("Method:", method)

            if viz:
                viz_path = scene_dir + f'/{method}_viz_{mask_type}_{split}/'
                os.makedirs(viz_path, exist_ok=True)

            img_eval_dict[scene_dir][method] = {}
            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"

            if not renders_dir.exists():
                raise FileNotFoundError(f"[metrics] renders not found: {renders_dir}")
            if not gt_dir.exists():
                raise FileNotFoundError(f"[metrics] gt not found: {gt_dir}")

            renders, gts, image_names = readImages(renders_dir, gt_dir)

            if mask_type in ["seen_mask", "mask", "eroded_seen_mask"]:
                masks_dir = method_dir / "masks"
                if not masks_dir.exists():
                    raise FileNotFoundError(f"[metrics] masks not found: {masks_dir}")
                masks, _ = readMasks(masks_dir)

                img_psnrs, img_ssims, img_lpipss = [], [], []
                psnrs_, sses_, num_valid_pixels = [], [], []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    sse_val = sse(renders[idx], gts[idx])
                    sses_.append(sse_val)

                    img_psnr = psnr(renders[idx], gts[idx])
                    img_ssim = ssim(renders[idx], gts[idx])
                    img_lpips = lpips(renders[idx], gts[idx])

                    valid_pix = masks[idx].squeeze().sum()

                    # 你原来的规则保留
                    if valid_pix > masks[idx].squeeze().numel() / 4.:
                        img_psnrs.append(img_psnr)
                        img_ssims.append(img_ssim)
                        img_lpipss.append(img_lpips)
                    else:
                        img_psnr = torch.tensor(-1, device="cuda")
                        img_ssim = torch.tensor(-1, device="cuda")
                        img_lpips = torch.tensor(-1, device="cuda")

                    if valid_pix > 0:
                        num_valid_pixels.append(valid_pix)
                        psnr_masked = 20 * torch.log10(1.0 / torch.sqrt(sse_val / valid_pix + 1e-5))
                        psnrs_.append(psnr_masked)

                    if viz and valid_pix > 0:
                        mse = (sse_val / valid_pix).item()
                        mse = "{:.4f}".format(mse)
                        render_np = renders[idx].squeeze().permute(1, 2, 0).cpu().numpy()
                        gt_np = gts[idx].squeeze().permute(1, 2, 0).cpu().numpy()
                        err_img = (((renders[idx] - gts[idx])) ** 2).squeeze().permute((1, 2, 0)).cpu().numpy()

                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        axs[0].imshow(render_np); axs[0].set_title(f'render\n mse:{mse}\n psnr:{img_psnr.item()}\n ssim:{img_ssim.item()}\n lpips:{img_lpips.item()}'); axs[0].axis('off')
                        axs[1].imshow(gt_np); axs[1].set_title('gt'); axs[1].axis('off')
                        axs[2].imshow(err_img); axs[2].set_title('err_img'); axs[2].axis('off')
                        fig.tight_layout()
                        plt.savefig(viz_path + image_names[idx])
                        plt.close()

                if len(psnrs_) == 0:
                    raise RuntimeError("[metrics] No valid pixels found in masks -> psnr list empty")

                print("  individual PSNR : {:>12.7f}".format(torch.tensor(psnrs_).mean(), ".5"))

                total_mse = torch.tensor(sses_).sum() / torch.tensor(num_valid_pixels).sum()
                total_psnr = 20 * torch.log10(1.0 / torch.sqrt(total_mse))
                print("  total PSNR : {:>12.7f}".format(total_psnr, ".5"))
                print("")

                full_dict[scene_dir][method].update({"PSNR": total_psnr.item()})
                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: v for v, name in zip(torch.tensor(psnrs_).tolist(), image_names)},
                    "Num Valid Pixels": {name: v for v, name in zip(torch.tensor(num_valid_pixels).tolist(), image_names)}
                })

                avg_img_psnr = torch.tensor(img_psnrs).mean() if len(img_psnrs) else torch.tensor(-1.0)
                avg_img_ssim = torch.tensor(img_ssims).mean() if len(img_ssims) else torch.tensor(-1.0)
                avg_img_lpips = torch.tensor(img_lpipss).mean() if len(img_lpipss) else torch.tensor(-1.0)

                print("Image PSNR: ", avg_img_psnr.item())
                print("Image SSIM: ", avg_img_ssim.item())
                print("Image LPIPS: ", avg_img_lpips.item())

                img_eval_dict[scene_dir][method].update({
                    "PSNR": avg_img_psnr.item(),
                    "SSIM": avg_img_ssim.item(),
                    "LPIPS": avg_img_lpips.item()
                })

            else:
                ssims_, psnrs_, lpipss_ = [], [], []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims_.append(ssim(renders[idx], gts[idx]))
                    psnrs_.append(psnr(renders[idx], gts[idx]))
                    lpipss_.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims_).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs_).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss_).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims_).mean().item(),
                    "PSNR": torch.tensor(psnrs_).mean().item(),
                    "LPIPS": torch.tensor(lpipss_).mean().item()
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: v for v, name in zip(torch.tensor(ssims_).tolist(), image_names)},
                    "PSNR": {name: v for v, name in zip(torch.tensor(psnrs_).tolist(), image_names)},
                    "LPIPS": {name: v for v, name in zip(torch.tensor(lpipss_).tolist(), image_names)}
                })

            with open(scene_dir + "/" + method + f"_img_eval_results{suffix}.json", "w") as fp:
                json.dump(img_eval_dict[scene_dir], fp, indent=True)

            with open(scene_dir + "/" + method + f"_results{suffix}.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)

            with open(scene_dir + "/" + method + f"_per_view{suffix}.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--iterations", type=int, default=2000)

    # ✅ new
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="evaluate which split folder (test or train)")

    parser.add_argument('--no_mask', action="store_true")
    parser.add_argument('--mask', action="store_true")
    parser.add_argument('--seen_mask', action="store_true")
    parser.add_argument('--eroded_seen_mask', action="store_true")
    parser.add_argument('--viz', action="store_true")
    args = parser.parse_args()

    mask_types = []
    if args.no_mask:
        mask_types.append("None")
    if args.mask:
        mask_types.append("mask")
    if args.eroded_seen_mask:
        mask_types.append("eroded_seen_mask")
    if args.seen_mask:
        mask_types.append("seen_mask")
    if len(mask_types) == 0:
        raise RuntimeError('Please specify the evaluation mode --[no_mask | mask | seen_mask]')

    for mask_type in mask_types:
        evaluate(args.model_paths, mask_type, split=args.split, viz=args.viz)
