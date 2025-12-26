"""Project a LiDAR point cloud (.pcd) into an image to build a depth map and save it as a PNG.

This is a minimal utility for debugging calibration / correspondence:
- You provide an image path and a matching LiDAR .pcd path
- Intrinsics K and extrinsics T_cam_lidar are hard-coded in this file (edit them)
- The point cloud is projected into the image to produce a sparse Z-depth map
- A viewable depth visualization PNG is written to disk

Usage:
  python scripts/project_lidar_pcd_to_depth_png.py \
    --image_path /path/to/frame.png \
    --pcd_path /path/to/lidar.pcd \
    --output_path /path/to/depth_vis.png

This will also write a 16-bit millimeter depth image (depth_mm.png) next to
--output_path unless you override it with --output_depth_mm_path.

It can also write a copy of the input image with projected pixels colored by
LiDAR intensity (depth_overlay.png) for easier visual correspondence.

Notes:
- Output is a visualization (8-bit) where nearer pixels are brighter.
- Unobserved pixels are 0 (black).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running this script from anywhere (without installing the package).
try:
    from mapanything.utils.lidar_projection import (
        load_pcd_xyzi,
        load_pcd_xyz,
        project_lidar_to_depth_z,
        project_lidar_to_depth_z_and_attribute,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    # If executed outside the repo root, Python may not find the local package.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from mapanything.utils.lidar_projection import (
        load_pcd_xyzi,
        load_pcd_xyz,
        project_lidar_to_depth_z,
        project_lidar_to_depth_z_and_attribute,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project LiDAR .pcd to a depth PNG")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--pcd_path", type=str, required=True)
    p.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Either an output PNG file path (e.g. /tmp/depth_vis.png) or an output directory (e.g. /tmp/projection/).",
    )
    p.add_argument(
        "--output_depth_mm_path",
        type=str,
        default=None,
        help="Optional path for 16-bit depth (millimeters) PNG. Default: sibling 'depth_mm.png' next to --output_path.",
    )
    p.add_argument(
        "--output_overlay_path",
        type=str,
        default=None,
        help="Optional path for an intensity-colored overlay image PNG. Default: sibling 'depth_overlay.png' next to --output_path.",
    )
    p.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.85,
        help="Overlay strength in [0,1]. 0 keeps original image, 1 replaces pixels with colormap colors.",
    )

    p.add_argument(
        "--overlay_color_by",
        type=str,
        default="intensity",
        choices=["intensity", "depth"],
        help="Which value to colorize the overlay with.",
    )
    p.add_argument(
        "--intensity_min",
        type=float,
        default=None,
        help="Min intensity for colormap scaling. If omitted, computed from percentiles.",
    )
    p.add_argument(
        "--intensity_max",
        type=float,
        default=None,
        help="Max intensity for colormap scaling. If omitted, computed from percentiles.",
    )
    p.add_argument(
        "--intensity_percentile_min",
        type=float,
        default=1.0,
        help="Lower percentile for auto intensity scaling (only used if intensity_min/max are omitted).",
    )
    p.add_argument(
        "--intensity_percentile_max",
        type=float,
        default=99.0,
        help="Upper percentile for auto intensity scaling (only used if intensity_min/max are omitted).",
    )
    p.add_argument("--min_depth", type=float, default=0.1)
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument(
        "--vis_max_depth",
        type=float,
        default=None,
        help="Max depth for visualization mapping. Defaults to --max_depth.",
    )
    return p


def _depth_to_vis_png(depth: np.ndarray, *, min_depth: float, max_depth: float) -> Image.Image:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {d.shape}")

    valid = d > 0
    d_clip = np.clip(d, float(min_depth), float(max_depth))

    vis = np.zeros_like(d_clip, dtype=np.uint8)
    if np.any(valid):
        vis[valid] = np.round(
            (1.0 - (d_clip[valid] - float(min_depth)) / (float(max_depth) - float(min_depth)))
            * 255.0
        ).astype(np.uint8)

    return Image.fromarray(vis, mode="L")


def _depth_to_mm_png(depth: np.ndarray, *, max_mm: int = 65535) -> Image.Image:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth must be (H,W), got {d.shape}")

    mm = np.zeros_like(d, dtype=np.uint16)
    valid = d > 0
    if np.any(valid):
        mm_f = np.round(d[valid] * 1000.0)
        mm_f = np.clip(mm_f, 0.0, float(max_mm))
        mm[valid] = mm_f.astype(np.uint16)

    # Pillow writes 16-bit PNG from uint16 arrays.
    return Image.fromarray(mm, mode="I;16")


def _scalar_to_colormap_rgb(values: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    """Convert a scalar image (H,W) to an RGB colormap (uint8) with a vivid jet-like palette.

    Pixels where values<=0 are treated as unobserved and set to (0,0,0).
    """

    v = np.asarray(values, dtype=np.float32)
    if v.ndim != 2:
        raise ValueError(f"values must be (H,W), got {v.shape}")

    if float(vmax) <= float(vmin):
        raise ValueError("vmax must be > vmin")

    valid = v > 0
    t = np.zeros_like(v, dtype=np.float32)
    if np.any(valid):
        v_clip = np.clip(v[valid], float(vmin), float(vmax))
        t[valid] = (v_clip - float(vmin)) / (float(vmax) - float(vmin))

    # Jet-like colormap: vivid and commonly used.
    # r,g,b in [0,1]
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)

    rgb = np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.round(r * 255.0).astype(np.uint8)
    rgb[..., 1] = np.round(g * 255.0).astype(np.uint8)
    rgb[..., 2] = np.round(b * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _overlay_color_on_image(
    image_rgb: np.ndarray,
    scalar: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    alpha: float,
) -> np.ndarray:
    """Overlay colormapped scalar pixels onto the original image.

    Only pixels with scalar>0 are affected.
    """

    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image_rgb must be (H,W,3), got {img.shape}")

    s = np.asarray(scalar, dtype=np.float32)
    if s.shape != img.shape[:2]:
        raise ValueError(f"scalar shape {s.shape} must match image spatial shape {img.shape[:2]}")

    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise ValueError("overlay_alpha must be in [0,1]")

    colors = _scalar_to_colormap_rgb(s, vmin=vmin, vmax=vmax)
    valid = s > 0

    out = img.astype(np.float32).copy()
    if np.any(valid):
        out[valid] = (1.0 - a) * out[valid] + a * colors[valid].astype(np.float32)

    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)


def main() -> None:
    args = _build_parser().parse_args()

    image_path = Path(args.image_path)
    pcd_path = Path(args.pcd_path)
    output_path = Path(args.output_path)

    # Support --output_path as either a file or a directory.
    # If it has no suffix (e.g. "projection"), treat it as a directory.
    if output_path.suffix.lower() == "" or (output_path.exists() and output_path.is_dir()):
        out_dir = output_path
        out_vis_path = out_dir / "depth_vis.png"
    else:
        out_vis_path = output_path
        out_dir = out_vis_path.parent

    depth_mm_path = (
        Path(args.output_depth_mm_path)
        if args.output_depth_mm_path is not None
        else out_vis_path.with_name("depth_mm.png")
    )
    overlay_path = (
        Path(args.output_overlay_path)
        if args.output_overlay_path is not None
        else out_vis_path.with_name("depth_overlay.png")
    )

    if not image_path.exists():
        raise FileNotFoundError(str(image_path))
    if not pcd_path.exists():
        raise FileNotFoundError(str(pcd_path))

    # ---------------------------------------------------------------------
    # Hard-coded calibration (EDIT THESE)
    # ---------------------------------------------------------------------
    # Camera intrinsics (pixel coordinates) as 3x3:
    #   [[fx, 0, cx],
    #    [0, fy, cy],
    #    [0,  0,  1]]
    K = np.array(
        [
            [907.359497070312, 0.0, 632.73291015625],
            [0.0, 907.26708984375, 353.749420166016],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # LiDAR -> Camera transform (T_cam_lidar) as 4x4:
    # T_cam_lidar = np.eye(4, dtype=np.float32)
    T_cam_lidar = np.array(
        [
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 1.0, 0.0, 0.03],
            [0.0, 0.0, 1.0, -0.003],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # ---------------------------------------------------------------------

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_np = np.array(img, dtype=np.uint8)

    pts_lidar: np.ndarray
    intensity: np.ndarray | None = None
    try:
        pts_xyzi = load_pcd_xyzi(pcd_path)
        pts_lidar = pts_xyzi[:, :3]
        intensity = pts_xyzi[:, 3]
    except Exception:
        pts_lidar = load_pcd_xyz(pcd_path)

    if args.overlay_color_by == "intensity" and intensity is not None:
        depth, intensity_img = project_lidar_to_depth_z_and_attribute(
            pts_lidar,
            intensity,
            T_cam_lidar=T_cam_lidar,
            K=K,
            image_size_hw=(H, W),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
    else:
        intensity_img = None
        depth = project_lidar_to_depth_z(
            pts_lidar,
            T_cam_lidar=T_cam_lidar,
            K=K,
            image_size_hw=(H, W),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )

    vis_max = args.max_depth if args.vis_max_depth is None else float(args.vis_max_depth)
    depth_vis = _depth_to_vis_png(depth, min_depth=args.min_depth, max_depth=vis_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    depth_vis.save(out_vis_path)

    # Also save a metric depth PNG in millimeters (uint16). Unobserved pixels are 0.
    depth_mm_path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm_img = _depth_to_mm_png(depth)
    depth_mm_img.save(depth_mm_path)

    # Save an RGB overlay image with projected pixels colored by the requested value.
    if args.overlay_color_by == "intensity" and intensity_img is not None:
        observed = depth > 0
        inten_vals = intensity_img[observed]
        if args.intensity_min is not None and args.intensity_max is not None:
            imin, imax = float(args.intensity_min), float(args.intensity_max)
        elif inten_vals.size > 0:
            p0 = float(args.intensity_percentile_min)
            p1 = float(args.intensity_percentile_max)
            imin = float(np.percentile(inten_vals, p0))
            imax = float(np.percentile(inten_vals, p1))
        else:
            imin, imax = 0.0, 1.0

        # Use depth mask as observation mask; allow intensity to be 0 as a valid value.
        scalar_for_overlay = intensity_img.copy().astype(np.float32)
        scalar_for_overlay[~observed] = 0.0
        overlay_np = _overlay_color_on_image(
            img_np,
            scalar_for_overlay,
            vmin=imin,
            vmax=imax,
            alpha=args.overlay_alpha,
        )
        overlay_label = f"intensity (vmin={imin:.3g}, vmax={imax:.3g})"
    else:
        overlay_np = _overlay_color_on_image(
            img_np,
            depth,
            vmin=float(args.min_depth),
            vmax=float(vis_max),
            alpha=args.overlay_alpha,
        )
        overlay_label = f"depth (vmin={float(args.min_depth):.3g}, vmax={float(vis_max):.3g})"

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_np, mode="RGB").save(overlay_path)

    nonzero = int((depth > 0).sum())
    print(f"Saved depth visualization: {out_vis_path}")
    print(f"Saved depth (mm, 16-bit): {depth_mm_path}")
    print(f"Saved overlay image ({overlay_label}): {overlay_path}")
    print(f"Depth stats: nonzero_pixels={nonzero}, min={float(depth[depth>0].min()) if nonzero else 'n/a'}, max={float(depth.max())}")


if __name__ == "__main__":
    main()
