from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np


def extract_numeric_suffix(stem: str) -> int | None:
    """Extract the last run of digits from a filename stem.

    Examples:
      image1 -> 1
      lidar_00042 -> 42
      frame12_left -> 12
    """

    m = re.search(r"(\d+)(?!.*\d)", stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_pcd_header(lines: list[bytes]) -> tuple[dict[str, str], int]:
    """Parse PCD header from the beginning of a file.

    Returns (header_dict, header_line_count).
    """

    header: dict[str, str] = {}
    header_lines = 0
    for raw in lines:
        line = raw.decode("utf-8", errors="ignore").strip()
        header_lines += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        if key == "DATA":
            header[key] = " ".join(parts[1:]).lower()
            break
        header[key] = " ".join(parts[1:])
    return header, header_lines


def _pcd_expand_fields(fields: list[str], counts: list[int]) -> list[str]:
    expanded: list[str] = []
    for name, cnt in zip(fields, counts, strict=False):
        cnt_i = int(cnt)
        for j in range(cnt_i):
            expanded.append(f"{name}_{j}" if cnt_i > 1 else name)
    return expanded


def _pcd_find_first_field_index(name_to_idx: dict[str, int], candidates: tuple[str, ...]) -> int | None:
    for c in candidates:
        if c in name_to_idx:
            return int(name_to_idx[c])
    return None


def load_pcd_xyz(path: str | Path) -> np.ndarray:
    """Load x,y,z from a PCD file.

    Supports: DATA ascii, DATA binary.
    Does NOT support: binary_compressed.
    """

    p = Path(path)
    with p.open("rb") as f:
        raw_lines: list[bytes] = []
        for _ in range(200):
            raw = f.readline()
            if not raw:
                break
            raw_lines.append(raw)
            if raw.strip().lower().startswith(b"data "):
                break

        header, header_line_count = parse_pcd_header(raw_lines)
        data_kind = header.get("DATA", "").lower()
        if data_kind not in ("ascii", "binary"):
            raise ValueError(
                f"Unsupported PCD DATA type '{data_kind}' in {p}. Supported: ascii, binary"
            )

        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields:
            raise ValueError(f"PCD missing FIELDS in {p}")
        fields_list = fields.split()

        sizes = [int(x) for x in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts_raw = header.get("COUNT", "")
        counts = [int(x) for x in counts_raw.split()] if counts_raw else [1] * len(fields_list)

        if not (len(sizes) == len(types) == len(fields_list) == len(counts)):
            raise ValueError(
                f"PCD header mismatch in {p}: FIELDS/SIZE/TYPE/COUNT lengths differ"
            )

        if "POINTS" in header:
            n_points = int(header["POINTS"].split()[0])
        elif "WIDTH" in header and "HEIGHT" in header:
            n_points = int(header["WIDTH"].split()[0]) * int(header["HEIGHT"].split()[0])
        else:
            raise ValueError(f"PCD missing POINTS (or WIDTH/HEIGHT) in {p}")

        expanded_fields = _pcd_expand_fields(fields_list, counts)
        name_to_idx = {name: idx for idx, name in enumerate(expanded_fields)}
        if "x" not in name_to_idx or "y" not in name_to_idx or "z" not in name_to_idx:
            raise ValueError(f"PCD must contain x y z fields: {p}")
        ix, iy, iz = name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]

        # Seek to data section
        f.seek(0)
        for _ in range(header_line_count):
            f.readline()

        if data_kind == "ascii":
            text = f.read().decode("utf-8", errors="ignore")
            if not text.strip():
                return np.zeros((0, 3), dtype=np.float32)
            arr = np.loadtxt(text.splitlines(), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[0] < n_points:
                raise ValueError(
                    f"PCD ascii points mismatch: expected {n_points}, got {arr.shape[0]} ({p})"
                )
            arr = arr[:n_points]
            return np.stack([arr[:, ix], arr[:, iy], arr[:, iz]], axis=1).astype(np.float32)

        # binary
        numpy_types = []
        for field_name, size, typ, cnt in zip(fields_list, sizes, types, counts, strict=False):
            if typ.upper() == "F":
                base_dtype = np.dtype("<f4" if size == 4 else "<f8")
            elif typ.upper() == "I":
                base_dtype = np.dtype(
                    "<i1" if size == 1 else "<i2" if size == 2 else "<i4" if size == 4 else "<i8"
                )
            elif typ.upper() == "U":
                base_dtype = np.dtype(
                    "<u1" if size == 1 else "<u2" if size == 2 else "<u4" if size == 4 else "<u8"
                )
            else:
                raise ValueError(f"Unsupported PCD TYPE '{typ}' in {p}")

            cnt_i = int(cnt)
            if cnt_i == 1:
                numpy_types.append((field_name, base_dtype))
            else:
                for j in range(cnt_i):
                    numpy_types.append((f"{field_name}_{j}", base_dtype))

        dtype = np.dtype(numpy_types)
        raw = f.read(n_points * dtype.itemsize)
        if len(raw) < n_points * dtype.itemsize:
            raise ValueError(f"PCD binary payload too short in {p}")
        data = np.frombuffer(raw, dtype=dtype, count=n_points)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        z = data["z"].astype(np.float32)
        return np.stack([x, y, z], axis=1)


def load_pcd_xyzi(
    path: str | Path,
    *,
    intensity_field_candidates: tuple[str, ...] = (
        "intensity",
        "reflectance",
        "i",
        "intensities",
        "remission",
    ),
) -> np.ndarray:
    """Load x,y,z,intensity from a PCD file.

    The intensity field name varies by dataset/vendor; this function searches for
    common alternatives via `intensity_field_candidates`.

    Supports: DATA ascii, DATA binary.
    Does NOT support: binary_compressed.
    """

    p = Path(path)
    with p.open("rb") as f:
        raw_lines: list[bytes] = []
        for _ in range(200):
            raw = f.readline()
            if not raw:
                break
            raw_lines.append(raw)
            if raw.strip().lower().startswith(b"data "):
                break

        header, header_line_count = parse_pcd_header(raw_lines)
        data_kind = header.get("DATA", "").lower()
        if data_kind not in ("ascii", "binary"):
            raise ValueError(
                f"Unsupported PCD DATA type '{data_kind}' in {p}. Supported: ascii, binary"
            )

        fields = header.get("FIELDS") or header.get("FIELD")
        if not fields:
            raise ValueError(f"PCD missing FIELDS in {p}")
        fields_list = fields.split()

        sizes = [int(x) for x in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts_raw = header.get("COUNT", "")
        counts = [int(x) for x in counts_raw.split()] if counts_raw else [1] * len(fields_list)

        if not (len(sizes) == len(types) == len(fields_list) == len(counts)):
            raise ValueError(
                f"PCD header mismatch in {p}: FIELDS/SIZE/TYPE/COUNT lengths differ"
            )

        if "POINTS" in header:
            n_points = int(header["POINTS"].split()[0])
        elif "WIDTH" in header and "HEIGHT" in header:
            n_points = int(header["WIDTH"].split()[0]) * int(header["HEIGHT"].split()[0])
        else:
            raise ValueError(f"PCD missing POINTS (or WIDTH/HEIGHT) in {p}")

        expanded_fields = _pcd_expand_fields(fields_list, counts)
        name_to_idx = {name: idx for idx, name in enumerate(expanded_fields)}
        if "x" not in name_to_idx or "y" not in name_to_idx or "z" not in name_to_idx:
            raise ValueError(f"PCD must contain x y z fields: {p}")
        ix, iy, iz = name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]

        ii = _pcd_find_first_field_index(name_to_idx, intensity_field_candidates)
        if ii is None:
            raise ValueError(
                f"PCD missing an intensity-like field in {p}. Tried: {intensity_field_candidates}"
            )

        # Seek to data section
        f.seek(0)
        for _ in range(header_line_count):
            f.readline()

        if data_kind == "ascii":
            text = f.read().decode("utf-8", errors="ignore")
            if not text.strip():
                return np.zeros((0, 4), dtype=np.float32)
            arr = np.loadtxt(text.splitlines(), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[0] < n_points:
                raise ValueError(
                    f"PCD ascii points mismatch: expected {n_points}, got {arr.shape[0]} ({p})"
                )
            arr = arr[:n_points]
            return np.stack([arr[:, ix], arr[:, iy], arr[:, iz], arr[:, ii]], axis=1).astype(
                np.float32
            )

        # binary
        numpy_types = []
        for field_name, size, typ, cnt in zip(fields_list, sizes, types, counts, strict=False):
            if typ.upper() == "F":
                base_dtype = np.dtype("<f4" if size == 4 else "<f8")
            elif typ.upper() == "I":
                base_dtype = np.dtype(
                    "<i1"
                    if size == 1
                    else "<i2"
                    if size == 2
                    else "<i4"
                    if size == 4
                    else "<i8"
                )
            elif typ.upper() == "U":
                base_dtype = np.dtype(
                    "<u1"
                    if size == 1
                    else "<u2"
                    if size == 2
                    else "<u4"
                    if size == 4
                    else "<u8"
                )
            else:
                raise ValueError(f"Unsupported PCD TYPE '{typ}' in {p}")

            cnt_i = int(cnt)
            if cnt_i == 1:
                numpy_types.append((field_name, base_dtype))
            else:
                for j in range(cnt_i):
                    numpy_types.append((f"{field_name}_{j}", base_dtype))

        dtype = np.dtype(numpy_types)
        raw = f.read(n_points * dtype.itemsize)
        if len(raw) < n_points * dtype.itemsize:
            raise ValueError(f"PCD binary payload too short in {p}")
        data = np.frombuffer(raw, dtype=dtype, count=n_points)

        # Map back to the candidate field name that exists in the binary dtype.
        intensity_name: str | None = None
        for c in intensity_field_candidates:
            if c in data.dtype.names:
                intensity_name = c
                break
        if intensity_name is None:
            raise ValueError(
                f"PCD missing an intensity-like field in {p}. Tried: {intensity_field_candidates}"
            )

        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        z = data["z"].astype(np.float32)
        inten = data[intensity_name].astype(np.float32)
        return np.stack([x, y, z, inten], axis=1)


def project_lidar_to_depth_z_and_attribute(
    points_lidar: np.ndarray,
    attribute: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
    image_size_hw: Tuple[int, int],
    *,
    min_depth: float = 0.1,
    max_depth: float = 200.0,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points into the image to build a Z-depth map and attribute map.

    Returns (depth_z, attr) as (H,W) float32.
    Unobserved pixels are 0 in both outputs.

    Z-buffer behavior: if multiple points hit the same pixel, keeps the nearest
    (smallest Z), and uses its attribute value for the attribute map.
    """

    H, W = image_size_hw
    pts = np.asarray(points_lidar, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_lidar must be (N,3), got {pts.shape}")

    attr = np.asarray(attribute, dtype=np.float32).reshape(-1)
    if attr.shape[0] != pts.shape[0]:
        raise ValueError(
            f"attribute must have same length as points_lidar: {attr.shape[0]} vs {pts.shape[0]}"
        )

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        attr = attr[idx]

    T = np.asarray(T_cam_lidar, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_lidar must be (4,4), got {T.shape}")

    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam_h = (T @ pts_h.T).T
    x, y, z = pts_cam_h[:, 0], pts_cam_h[:, 1], pts_cam_h[:, 2]

    valid = (z > float(min_depth)) & (z < float(max_depth))
    if not np.any(valid):
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    x = x[valid]
    y = y[valid]
    z = z[valid]
    a = attr[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]
    a = a[in_bounds]

    if ui.size == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    depth = np.zeros((H, W), dtype=np.float32)
    attr_img = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    for px, py, pz, pa in zip(ui, vi, z, a, strict=False):
        if pz < zbuf[py, px]:
            zbuf[py, px] = pz
            attr_img[py, px] = float(pa)

    mask = np.isfinite(zbuf)
    depth[mask] = zbuf[mask]
    return depth, attr_img


def project_lidar_to_depth_z(
    points_lidar: np.ndarray,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
    image_size_hw: Tuple[int, int],
    *,
    min_depth: float = 0.1,
    max_depth: float = 200.0,
    max_points: int | None = None,
) -> np.ndarray:
    """Project LiDAR points (LiDAR frame) into the image to build a Z-depth map.

    Returns depth_z as (H,W) float32. Unobserved pixels are 0.

    Z-buffer behavior: if multiple points hit the same pixel, keeps the nearest (smallest Z).
    """

    H, W = image_size_hw
    pts = np.asarray(points_lidar, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_lidar must be (N,3), got {pts.shape}")

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    T = np.asarray(T_cam_lidar, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_lidar must be (4,4), got {T.shape}")

    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    # Transform to camera frame
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # (N,4)
    pts_cam_h = (T @ pts_h.T).T  # (N,4)
    x, y, z = pts_cam_h[:, 0], pts_cam_h[:, 1], pts_cam_h[:, 2]

    valid = (z > float(min_depth)) & (z < float(max_depth))
    if not np.any(valid):
        return np.zeros((H, W), dtype=np.float32)

    x = x[valid]
    y = y[valid]
    z = z[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]

    if ui.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    depth = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    for px, py, pz in zip(ui, vi, z, strict=False):
        if pz < zbuf[py, px]:
            zbuf[py, px] = pz

    mask = np.isfinite(zbuf)
    depth[mask] = zbuf[mask]
    return depth
