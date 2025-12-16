import os
import shutil
from pathlib import Path

def read_images_txt(path: Path):
    """
    COLMAP images.txt: two-line per image (header + points2D)
    Return list of tuples: (image_id:int, header_line:str, points_line:str, image_name:str)
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    items = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (not line) or line.startswith("#"):
            i += 1
            continue
        header = lines[i]
        pts = lines[i+1] if i + 1 < len(lines) else ""
        parts = header.split()
        # header: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID IMAGE_NAME
        image_id = int(parts[0])
        image_name = parts[9]
        items.append((image_id, header, pts, image_name))
        i += 2
    return items

def write_images_txt(path: Path, kept_items):
    with path.open("w", encoding="utf-8") as f:
        f.write("# Downsampled images.txt\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for (image_id, header, pts, image_name) in kept_items:
            f.write(header.rstrip() + "\n")
            f.write(pts.rstrip() + "\n")

def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="input dataset root, e.g. scene0207_00")
    ap.add_argument("--out_root", required=True, help="output dataset root, e.g. scene0207_00_sub10")
    ap.add_argument("--k", type=int, required=True, help="keep 1 every K frames, K>=1")
    ap.add_argument("--images_dir", default="images", help="images folder name under root")
    ap.add_argument("--depth_dir", default=None, help="depth folder name under root (e.g. depth or depths). If omitted, will auto-detect.")
    ap.add_argument("--keep_pose", action="store_true", help="also copy pose folder if exists")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    k = max(1, int(args.k))

    sparse0 = in_root / "sparse" / "0"
    images_txt = sparse0 / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(f"Missing {images_txt}")

    # auto-detect depth folder
    depth_dir = args.depth_dir
    if depth_dir is None:
        if (in_root / "depths").is_dir():
            depth_dir = "depths"
        elif (in_root / "depth").is_dir():
            depth_dir = "depth"
        else:
            depth_dir = None

    items = read_images_txt(images_txt)
    # sort by image_name to be deterministic
    items = sorted(items, key=lambda x: x[3])

    kept = items[::k]
    print(f"[Downsample] keep 1/{k}: {len(kept)}/{len(items)} images")

    # prepare out structure
    (out_root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (out_root / args.images_dir).mkdir(parents=True, exist_ok=True)
    if depth_dir:
        (out_root / depth_dir).mkdir(parents=True, exist_ok=True)

    # copy sparse/0 essentials (except images.bin)
    # cameras.* keep as-is
    copy_if_exists(sparse0 / "cameras.bin", out_root / "sparse" / "0" / "cameras.bin")
    copy_if_exists(sparse0 / "cameras.txt", out_root / "sparse" / "0" / "cameras.txt")
    # points3D.* keep as-is (ply/bin/txt whichever exists)
    for fn in ["points3D.ply", "points3D.bin", "points3D.txt"]:
        copy_if_exists(sparse0 / fn, out_root / "sparse" / "0" / fn)

    # write downsampled images.txt
    write_images_txt(out_root / "sparse" / "0" / "images.txt", kept)

    # IMPORTANT: ensure images.bin not present in out (loader prefers bin)
    out_images_bin = out_root / "sparse" / "0" / "images.bin"
    if out_images_bin.exists():
        out_images_bin.unlink()

    # copy actual image/depth files
    in_img_dir = in_root / args.images_dir
    if not in_img_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {in_img_dir}")

    in_depth_dir = (in_root / depth_dir) if depth_dir else None

    for _, _, _, name in kept:
        # COLMAP stores relative name like "frame0001.png" (sometimes with subfolders)
        src_img = in_img_dir / name
        dst_img = out_root / args.images_dir / name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image file: {src_img}")
        shutil.copy2(src_img, dst_img)

        if in_depth_dir is not None and in_depth_dir.is_dir():
            # depth often same basename but png; try exact match first
            src_depth = in_depth_dir / Path(name).with_suffix(".png").name
            if not src_depth.exists():
                # fallback: same relative path, just suffix png
                src_depth = (in_depth_dir / Path(name)).with_suffix(".png")
            if src_depth.exists():
                dst_depth = out_root / depth_dir / src_depth.name
                shutil.copy2(src_depth, dst_depth)

    # copy optional other files
    for fn in ["observations.txt", "pointcloud.ply", "trajectory.txt"]:
        copy_if_exists(in_root / fn, out_root / fn)

    if args.keep_pose and (in_root / "pose").is_dir():
        copy_if_exists(in_root / "pose", out_root / "pose")

    print("[Done] Output:", out_root)

if __name__ == "__main__":
    main()

