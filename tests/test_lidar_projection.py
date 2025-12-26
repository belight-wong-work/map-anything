import numpy as np
import pytest
from PIL import Image

from mapanything.utils.lidar_projection import (
    extract_numeric_suffix,
    load_pcd_xyzi,
    load_pcd_xyz,
    project_lidar_to_depth_z_and_attribute,
    project_lidar_to_depth_z,
)


def test_extract_numeric_suffix():
    assert extract_numeric_suffix("image1") == 1
    assert extract_numeric_suffix("image10") == 10
    assert extract_numeric_suffix("lidar_00042") == 42
    assert extract_numeric_suffix("frame12_left") == 12
    assert extract_numeric_suffix("no_digits") is None


def _write_pcd_ascii(path, points_xyz: np.ndarray):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {points_xyz.shape[0]}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {points_xyz.shape[0]}",
            "DATA ascii",
        ]
    )
    lines = [header]
    for x, y, z in points_xyz:
        lines.append(f"{x} {y} {z}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pcd_ascii_xyzi(path, points_xyzi: np.ndarray):
    points_xyzi = np.asarray(points_xyzi, dtype=np.float32)
    assert points_xyzi.ndim == 2 and points_xyzi.shape[1] == 4
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {points_xyzi.shape[0]}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {points_xyzi.shape[0]}",
            "DATA ascii",
        ]
    )
    lines = [header]
    for x, y, z, i in points_xyzi:
        lines.append(f"{x} {y} {z} {i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_pcd_xyz_ascii(tmp_path):
    pts = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 10.0],
        ],
        dtype=np.float32,
    )
    p = tmp_path / "lidar1.pcd"
    _write_pcd_ascii(p, pts)
    loaded = load_pcd_xyz(p)
    assert loaded.shape == (3, 3)
    np.testing.assert_allclose(loaded, pts, rtol=0, atol=1e-6)


def test_load_pcd_xyzi_ascii(tmp_path):
    pts = np.array(
        [
            [0.0, 0.0, 1.0, 10.0],
            [1.0, 2.0, 3.0, 20.0],
        ],
        dtype=np.float32,
    )
    p = tmp_path / "lidar_i.pcd"
    _write_pcd_ascii_xyzi(p, pts)
    loaded = load_pcd_xyzi(p)
    assert loaded.shape == (2, 4)
    np.testing.assert_allclose(loaded, pts, rtol=0, atol=1e-6)


def test_project_lidar_to_depth_z_and_attribute_zbuffer_keeps_nearest_attr():
    H, W = 80, 100
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    T = np.eye(4, dtype=np.float32)

    # Two points map to the same pixel (50,40) but different depths and intensities.
    pts = np.array(
        [
            [0.0, 0.0, 10.0],  # farther
            [0.0, 0.0, 5.0],  # nearer
        ],
        dtype=np.float32,
    )
    inten = np.array([1.0, 99.0], dtype=np.float32)

    depth, attr_img = project_lidar_to_depth_z_and_attribute(
        pts,
        inten,
        T_cam_lidar=T,
        K=K,
        image_size_hw=(H, W),
    )

    assert depth[40, 50] == pytest.approx(5.0)
    assert attr_img[40, 50] == pytest.approx(99.0)


def test_project_lidar_to_depth_z_basic_and_zbuffer():
    # Image size
    H, W = 100, 120

    # Intrinsics: principal point at (50, 40)
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    # Identity extrinsic
    T = np.eye(4, dtype=np.float32)

    # Points: all in front (z>0)
    # p0 projects to (50,40) with z=10
    # p1 projects to (60,40) with z=10
    # p2 projects to (50,50) with z=10
    # p3 projects to same pixel as p0 but closer (z=5) -> should win in z-buffer
    pts = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    depth = project_lidar_to_depth_z(pts, T_cam_lidar=T, K=K, image_size_hw=(H, W))
    assert depth.shape == (H, W)

    assert depth[40, 50] == pytest.approx(5.0)  # z-buffer kept nearer point
    assert depth[40, 60] == pytest.approx(10.0)
    assert depth[50, 50] == pytest.approx(10.0)


def test_project_lidar_to_depth_z_with_extrinsic_translation():
    H, W = 100, 120
    K = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    # Translate lidar points by +1m in camera X (lidar origin appears shifted right)
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = 1.0

    # Point at (0,0,10) in lidar becomes (1,0,10) in cam -> u = 100*(0.1)+50 = 60
    pts = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
    depth = project_lidar_to_depth_z(pts, T_cam_lidar=T, K=K, image_size_hw=(H, W))

    assert depth[40, 60] == pytest.approx(10.0)
    assert depth[40, 50] == pytest.approx(0.0)


def test_project_lidar_to_depth_z_e2e_saves_depth_png(tmp_path):
    # Create an example RGB image on disk (the algorithm only needs H,W).
    H, W = 48, 64
    img = (np.linspace(0, 255, num=H * W * 3, dtype=np.uint8).reshape(H, W, 3))
    image_path = tmp_path / "frame1.png"
    Image.fromarray(img, mode="RGB").save(image_path)

    # Create a small synthetic point cloud and write it as an ASCII .pcd on disk.
    # Points are defined in LiDAR frame; here we use identity extrinsics so LiDAR==Camera.
    # We design points to land inside the image.
    z = 10.0
    xs = np.linspace(-1.0, 1.0, num=9, dtype=np.float32)
    ys = np.linspace(-0.75, 0.75, num=7, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)
    pts = np.concatenate([grid, np.full((grid.shape[0], 1), z, dtype=np.float32)], axis=1)
    pcd_path = tmp_path / "lidar1.pcd"
    _write_pcd_ascii(pcd_path, pts)

    # Hard-coded intrinsics/extrinsics (as requested)
    K = np.array(
        [[40.0, 0.0, W / 2.0], [0.0, 40.0, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    T = np.eye(4, dtype=np.float32)  # T_cam_lidar

    # Read from paths, project, and save a depth visualization as PNG.
    img_loaded = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    pts_loaded = load_pcd_xyz(pcd_path)
    depth = project_lidar_to_depth_z(
        pts_loaded,
        T_cam_lidar=T,
        K=K,
        image_size_hw=(img_loaded.shape[0], img_loaded.shape[1]),
        min_depth=0.1,
        max_depth=200.0,
    )
    assert depth.shape == (H, W)
    assert int((depth > 0).sum()) > 0

    # Save a viewable depth PNG: map depth range to [0,255] (near = bright).
    min_d, max_d = 0.1, 50.0
    d = depth.copy()
    valid = d > 0
    d = np.clip(d, min_d, max_d)
    depth_vis = np.zeros_like(d, dtype=np.uint8)
    depth_vis[valid] = np.round((1.0 - (d[valid] - min_d) / (max_d - min_d)) * 255.0).astype(
        np.uint8
    )
    out_path = tmp_path / "depth_vis.png"
    Image.fromarray(depth_vis, mode="L").save(out_path)
    assert out_path.exists()

    # Also save a 16-bit millimeter depth PNG for exact depth inspection.
    depth_mm = np.zeros_like(depth, dtype=np.uint16)
    valid = depth > 0
    depth_mm[valid] = np.clip(np.round(depth[valid] * 1000.0), 0.0, 65535.0).astype(np.uint16)
    out_mm_path = tmp_path / "depth_mm.png"
    Image.fromarray(depth_mm, mode="I;16").save(out_mm_path)
    assert out_mm_path.exists()
