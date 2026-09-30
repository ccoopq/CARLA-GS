import os
import numpy as np
import imageio


def load_normal_from_png(path):
    """
    从 RGB normal png 读取并转换到 [-1,1]
    """
    rgb = imageio.imread(path).astype(np.float32)  # (H,W,3), range [0,255]
    normals = rgb / 255.0 * 2.0 - 1.0              # 映射到 [-1,1]
    return normals


def ransac_plane(points, num_iters=200, dist_thresh=0.05, min_inliers_ratio=0.3):
    """
    在3D点云上用RANSAC拟合平面
    Args:
        points: (N,3) numpy
    Returns:
        inlier_mask: (N,) bool, 是否为内点
        plane: (normal, d)
    """
    if len(points) < 3:
        return np.zeros(len(points), dtype=bool), None

    best_inliers = []
    best_plane = None
    for _ in range(num_iters):
        idx = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[idx]

        # 拟合平面
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p1)

        dist = np.abs(points @ normal + d)
        inliers = np.where(dist < dist_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)

    inlier_mask = np.zeros(len(points), dtype=bool)
    if best_plane is not None and len(best_inliers) > min_inliers_ratio * len(points):
        inlier_mask[best_inliers] = True

    return inlier_mask, best_plane


def ground_mask_from_normals_and_depth(normals, depth,
                                       angle_thresh_deg=20.0,
                                       ransac_iters=256,
                                       depth_thresh_ratio=0.01):
    """
    输入: normals (H,W,3), depth (H,W)
    输出: ground_mask (H,W) bool
    """
    H, W, _ = normals.shape
    n = normals.reshape(-1, 3).astype(np.float32)
    d = depth.reshape(-1)

    # 单位化
    norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    n = n / norm

    # 有效像素
    valid = np.isfinite(n).all(axis=1) & (np.linalg.norm(n, axis=1) > 0) & np.isfinite(d) & (d > 0)
    n_valid = n[valid]
    if len(n_valid) == 0:
        return np.zeros((H, W), dtype=bool)

    cos_thr = np.cos(np.deg2rad(angle_thresh_deg))

    # -------- Step 1: normal RANSAC ----------
    best_inliers = []
    best_dir = None
    for _ in range(ransac_iters):
        idx = np.random.choice(len(n_valid), 1)
        cand = n_valid[idx][0]
        dot = n_valid @ cand
        inliers = np.where(np.abs(dot) >= cos_thr)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_dir = cand

    if best_dir is None or len(best_inliers) < 3:
        return np.zeros((H, W), dtype=bool)

    # 初步mask（normal一致性）
    mask_stage1 = np.zeros(H * W, dtype=bool)
    mask_stage1[valid] = False
    valid_idx = np.where(valid)[0]
    mask_stage1[valid_idx[best_inliers]] = True

    # -------- Step 2: 在候选点上用 depth RANSAC ----------
    ys, xs = np.where(mask_stage1.reshape(H, W))
    zs = depth[ys, xs]
    pts3d = np.stack([xs, ys, zs], axis=-1).astype(np.float32)  # (N,3)

    # 用深度决定自适应阈值
    median_depth = np.median(zs)
    dist_thresh = depth_thresh_ratio * median_depth

    inliers_depth, plane = ransac_plane(pts3d, num_iters=200, dist_thresh=dist_thresh)

    final_mask = np.zeros(H * W, dtype=bool)
    if plane is not None:
        sel = (ys * W + xs)[inliers_depth]
        final_mask[sel] = True

    return final_mask.reshape(H, W)


def process_normals_and_depth(normal_dir, depth_dir, save_dir,
                              angle_thresh_deg=20.0, depth_thresh_ratio=0.01):
    os.makedirs(save_dir, exist_ok=True)
    file_list = [f for f in os.listdir(normal_dir) if f.endswith(".png")]

    for fname in file_list:
        normal_path = os.path.join(normal_dir, fname)
        depth_path = os.path.join(depth_dir, fname.replace(".png", ".npy"))  # depth.npy

        if not os.path.exists(depth_path):
            print(f"[Skip] Depth not found for {fname}")
            continue

        normals = load_normal_from_png(normal_path)
        depth = np.load(depth_path)

        ground_mask = ground_mask_from_normals_and_depth(
            normals, depth, angle_thresh_deg=angle_thresh_deg, depth_thresh_ratio=depth_thresh_ratio
        )

        mask_np = (ground_mask.astype(np.uint8) * 255)
        save_path = os.path.join(save_dir, fname)
        imageio.imwrite(save_path, mask_np)
        print(f"[OK] {fname} -> {save_path}")


if __name__ == "__main__":
    normal_dir = "../../data/waymo/training/002/gt_normal"
    depth_dir = "../../data/waymo/training/002/gt_depth"
    save_dir = "../../data/waymo/training/002/gt_ground_mask"
    process_normals_and_depth(normal_dir, depth_dir, save_dir,
                              angle_thresh_deg=20, depth_thresh_ratio=0.05)

