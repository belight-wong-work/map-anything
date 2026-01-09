# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""MapAnything demo: images + shared intrinsics K + per-frame LiDAR point clouds.

This script is for the common robotics setup:
- Each image frame has a time-aligned LiDAR scan / point cloud (in LiDAR frame)
- LiDAR↔Camera extrinsic is known and fixed
- Camera intrinsics K are known (shared across all frames)
- Camera poses (trajectory) are NOT available

We project LiDAR points into each image to build a *sparse* Z-depth map (depth_z)
for that frame, then run `MapAnything.infer()` with inputs:
  img + intrinsics + depth_z

Notes / caveats:
- No camera poses are provided; the model will estimate relative poses internally.
- The output world frame is an arbitrary model frame (but depth is metric if your
  LiDAR depth is metric).
- The sparse depth map has many empty pixels; you may want to increase LiDAR
  density or apply your own depth completion if needed.

Expected file layout (default pairing by sorted filenames):
  --image_folder: /path/images/  (jpg/png)
  --lidar_folder: /path/lidar/   (.npy or .npz point clouds)

For each i, image[i] pairs with lidar[i].

Point cloud formats:
- .npy: (N,3) float32/float64
- .npz: must contain array "points" with shape (N,3)

Usage:
  python scripts/demo_images_k_lidar_inference.py \
    --image_folder /path/images \
    --lidar_folder /path/lidar \
    --T_cam_lidar_path /path/T_cam_lidar.npy \
    --K_path /path/K.npy \
    --output_ply output.ply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
import trimesh

from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.lidar_projection import (
    extract_numeric_suffix,
    load_pcd_xyz,
    project_lidar_to_depth_z,
)


# ----------------------------------------------------------------------------
# Optional: hard-code calibration here (as you requested)
#
# If you set these overrides, you do NOT need to pass --K_path / --T_cam_lidar_path.
# ----------------------------------------------------------------------------

# Shared camera intrinsics for ALL frames, in ORIGINAL input image pixel coords.
K_OVERRIDE: np.ndarray | None = None
# Example:
K_OVERRIDE = np.array(
    [
        [907.359497070312, 0.0, 632.73291015625],
        [0.0, 907.26708984375, 353.749420166016],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# Fixed LiDAR->Camera transform (T_cam_lidar), 4x4.
T_CAM_LIDAR_OVERRIDE: np.ndarray | None = None
# Example:
T_CAM_LIDAR_OVERRIDE = np.array(
    [
        [1.0, 0.0, 0.0, 0.012],
        [0.0, 1.0, 0.0, 0.03],
        [0.0, 0.0, 1.0, -0.003],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

def _load_matrix_4x4(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".npy":
        mat = np.load(p)
    elif p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        mat = np.array(obj, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported matrix file: {p.suffix}. Use .npy or .json")

    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected a (4,4) matrix in {p}, got {mat.shape}")
    return mat


def _load_intrinsics_3x3(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".npy":
        k = np.load(p)
        k = np.asarray(k, dtype=np.float32)
        if k.shape != (3, 3):
            raise ValueError(f"Expected a (3,3) K matrix in {p}, got {k.shape}")
        return k

    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        # Support either a direct 3x3 list, or {fx,fy,cx,cy}
        if isinstance(obj, list):
            k = np.asarray(obj, dtype=np.float32)
            if k.shape != (3, 3):
                raise ValueError(f"Expected 3x3 list in {p}, got {k.shape}")
            return k
        if isinstance(obj, dict) and all(k in obj for k in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = (float(obj["fx"]), float(obj["fy"]), float(obj["cx"]), float(obj["cy"]))
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        raise ValueError(
            f"Unsupported intrinsics JSON in {p}. Use a 3x3 list or a dict with fx,fy,cx,cy"
        )

    raise ValueError(f"Unsupported intrinsics file: {p.suffix}. Use .npy or .json")


def _load_points(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        pts = np.load(path)
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected point array (N,3[+]), got {pts.shape} from {path}")
        return np.asarray(pts[:, :3], dtype=np.float32)
    if suf == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise ValueError(f"{path} is .npz but missing key 'points'")
        pts = np.asarray(data["points"])
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Expected point array (N,3[+]), got {pts.shape} from {path}")
        return np.asarray(pts[:, :3], dtype=np.float32)
    if suf == ".pcd":
        return load_pcd_xyz(path)

    raise ValueError(f"Unsupported LiDAR file {path.name}. Use .pcd/.npy/.npz")


def _list_images(folder: str) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png")
    p = Path(folder)
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])


def _list_lidar(folder: str) -> list[Path]:
    exts = (".pcd", ".npy", ".npz")
    p = Path(folder)
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])


def _index_paths_by_suffix(paths: list[Path], kind: str) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    skipped = 0
    for p in paths:
        idx = extract_numeric_suffix(p.stem)
        if idx is None:
            skipped += 1
            continue
        # If duplicates occur, keep the first one (stable) and warn.
        if idx in indexed:
            print(f"Warning: duplicate {kind} index {idx}: '{indexed[idx].name}' and '{p.name}'. Using '{indexed[idx].name}'.")
            continue
        indexed[idx] = p
    if skipped:
        print(f"Warning: skipped {skipped} {kind} files without numeric suffix")
    return indexed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MapAnything: images + K + LiDAR (no poses)")
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--lidar_folder", type=str, required=True)

    p.add_argument(
        "--T_cam_lidar_path",
        type=str,
        default=None,
        help="4x4 transform from LiDAR frame to Camera frame (T_cam_lidar), in .npy or .json. Optional if T_CAM_LIDAR_OVERRIDE is set.",
    )
    p.add_argument(
        "--T_is_lidar_from_cam",
        action="store_true",
        default=False,
        help="If set, the provided matrix is T_lidar_cam and will be inverted to get T_cam_lidar.",
    )

    p.add_argument(
        "--K_path",
        type=str,
        default=None,
        help="Camera intrinsics K (3x3) as .npy, or .json with fx/fy/cx/cy or 3x3 list. Optional if K_OVERRIDE is set.",
    )

    p.add_argument("--model", type=str, default="facebook/map-anything")
    p.add_argument("--apache", action="store_true", help="Use facebook/map-anything-apache")

    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_views", type=int, default=None)

    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=200.0)
    p.add_argument("--max_points", type=int, default=None, help="Subsample LiDAR points per frame for speed")

    p.add_argument("--memory_efficient_inference", action="store_true", default=False)
    p.add_argument("--no_amp", action="store_true", default=False)
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    p.add_argument("--output_ply", type=str, default="output.ply")
    p.add_argument("--output_glb", type=str, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    image_paths_all = _list_images(args.image_folder)
    lidar_paths_all = _list_lidar(args.lidar_folder)

    if len(image_paths_all) == 0:
        raise ValueError(f"No images found in {args.image_folder}")
    if len(lidar_paths_all) == 0:
        raise ValueError(f"No LiDAR files found in {args.lidar_folder}")

    # Pair by numeric suffix (e.g., image2.jpeg <-> lidar2.pcd).
    # This avoids lexicographic sorting issues like image10 coming before image2.
    image_by_idx = _index_paths_by_suffix(image_paths_all, kind="image")
    lidar_by_idx = _index_paths_by_suffix(lidar_paths_all, kind="lidar")

    common_indices = sorted(set(image_by_idx.keys()) & set(lidar_by_idx.keys()))
    if len(common_indices) == 0:
        raise ValueError(
            "No matching numeric suffix indices found between images and LiDAR. "
            "Expected names like image1.jpeg and lidar1.pcd with the same trailing number."
        )

    missing_images = sorted(set(lidar_by_idx.keys()) - set(image_by_idx.keys()))
    missing_lidar = sorted(set(image_by_idx.keys()) - set(lidar_by_idx.keys()))
    if missing_images:
        print(f"Warning: missing images for indices: {missing_images[:20]}{' ...' if len(missing_images) > 20 else ''}")
    if missing_lidar:
        print(f"Warning: missing lidar for indices: {missing_lidar[:20]}{' ...' if len(missing_lidar) > 20 else ''}")

    image_paths = [image_by_idx[i] for i in common_indices]
    lidar_paths = [lidar_by_idx[i] for i in common_indices]

    stride = max(int(args.stride), 1)
    image_paths = image_paths[::stride]
    lidar_paths = lidar_paths[::stride]

    if args.max_views is not None:
        max_views = max(int(args.max_views), 1)
        image_paths = image_paths[:max_views]
        lidar_paths = lidar_paths[:max_views]

    print(f"Using {len(image_paths)} paired frames")

    if T_CAM_LIDAR_OVERRIDE is not None:
        T_cam_lidar = np.asarray(T_CAM_LIDAR_OVERRIDE, dtype=np.float32)
        if T_cam_lidar.shape != (4, 4):
            raise ValueError(f"T_CAM_LIDAR_OVERRIDE must be (4,4), got {T_cam_lidar.shape}")
    else:
        if not args.T_cam_lidar_path:
            raise ValueError("Provide --T_cam_lidar_path or set T_CAM_LIDAR_OVERRIDE in the script")
        T = _load_matrix_4x4(args.T_cam_lidar_path)
        if args.T_is_lidar_from_cam:
            T = np.linalg.inv(T)
        T_cam_lidar = T

    if K_OVERRIDE is not None:
        K = np.asarray(K_OVERRIDE, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"K_OVERRIDE must be (3,3), got {K.shape}")
    else:
        if not args.K_path:
            raise ValueError("Provide --K_path or set K_OVERRIDE in the script")
        K = _load_intrinsics_3x3(args.K_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/map-anything-apache" if args.apache else args.model
    print("Device:", device)
    print("Model:", model_id)
    model = MapAnything.from_pretrained(model_id).to(device)

    raw_views = []
    for img_p, lidar_p in zip(image_paths, lidar_paths, strict=False):
        img = np.array(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        H, W = img.shape[0], img.shape[1]

        pts_lidar = _load_points(lidar_p)
        depth_z = project_lidar_to_depth_z(
            pts_lidar,
            T_cam_lidar=T_cam_lidar,
            K=K,
            image_size_hw=(H, W),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_points=args.max_points,
        )

        raw_views.append(
            {
                "img": img,
                "intrinsics": K,
                "depth_z": depth_z,
                # LiDAR is metric.
                "is_metric_scale": torch.tensor([True]),
            }
        )

    # This will resize/crop images to the model's working resolution and will update K/depth accordingly.
    views = preprocess_inputs(raw_views, norm_type="dinov2", resize_mode="fixed_mapping")

    print("Running inference...")
    outputs = model.infer(
        views,
        memory_efficient_inference=bool(args.memory_efficient_inference),
        use_amp=not bool(args.no_amp),
        amp_dtype=str(args.amp_dtype),
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
    )
    print("Inference complete")

    # Export a merged point cloud
    pts_all = []
    cols_all = []

    for pred in outputs:
        pts3d = pred["pts3d"][0].detach().cpu().numpy()  # (H,W,3)
        rgb = pred["img_no_norm"][0].detach().cpu().numpy()  # (H,W,3) float [0,1]
        mask = pred["mask"][0].squeeze(-1).detach().cpu().numpy().astype(bool)

        pts_all.append(pts3d[mask].reshape(-1, 3))
        cols_all.append((rgb[mask].reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8))

    pts_all_np = np.concatenate(pts_all, axis=0) if len(pts_all) else np.zeros((0, 3), dtype=np.float32)
    cols_all_np = np.concatenate(cols_all, axis=0) if len(cols_all) else np.zeros((0, 3), dtype=np.uint8)

    out_ply = Path(args.output_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    print("Writing PLY:", str(out_ply))
    trimesh.PointCloud(pts_all_np, colors=cols_all_np).export(str(out_ply))

    if args.output_glb:
        # Optional: export as GLB point cloud (trimesh supports glb export too)
        out_glb = Path(args.output_glb)
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        print("Writing GLB:", str(out_glb))
        trimesh.PointCloud(pts_all_np, colors=cols_all_np).export(str(out_glb))


if __name__ == "__main__":
    main()
