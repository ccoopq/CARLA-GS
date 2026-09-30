'''
python eval_nor_depth.py --split train
'''

import argparse
import json
import os
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate depth and normal errors for the waymo_train_002 model variants."
    )
    parser.add_argument("--config", default="configs/example/waymo_train_002.yaml")
    parser.add_argument("--output-root", default="output/waymo_full_exp")
    parser.add_argument("--exp-prefix", default="waymo_train_002")
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=["baseline", "flat", "normal", "geo"],
        help="Experiment suffixes under output-root, e.g. baseline flat normal geo.",
    )
    parser.add_argument(
        "--model-dirs",
        nargs="*",
        default=None,
        help="Explicit model directories. Overrides output-root/exp-prefix/suffixes.",
    )
    parser.add_argument("--iteration", type=int, default=50000)
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--max-views", type=int, default=-1)
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--min-acc", type=float, default=0.0)
    parser.add_argument(
        "--mask",
        choices=["valid", "ground"],
        default="valid",
        help="valid: all valid GT depth pixels. ground: valid GT pixels inside gt_ground_mask.",
    )
    parser.add_argument(
        "--raw-depth",
        action="store_true",
        help="Compare raw rasterized depth. Default compares expected depth = depth / alpha.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--per-view", action="store_true")

    eval_args, cfg_args = parser.parse_known_args()

    # lib.config parses sys.argv at import time. Keep only the arguments it knows
    # about, plus any trailing config overrides the caller supplied.
    sys.argv = [sys.argv[0], "--config", eval_args.config] + cfg_args
    return eval_args


ARGS = parse_args()

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lib.config import cfg
from lib.datasets.waymo_full_readers import readWaymoFullInfo
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.utils.camera_utils import cameraList_from_camInfos
from lib.utils.system_utils import searchForMaxIteration


def as_abs(path):
    return path if os.path.isabs(path) else os.path.abspath(path)


def resolve_model_dirs():
    if ARGS.model_dirs:
        return [as_abs(path) for path in ARGS.model_dirs]

    output_root = as_abs(ARGS.output_root)
    return [
        os.path.join(output_root, f"{ARGS.exp_prefix}-{suffix}")
        for suffix in ARGS.suffixes
    ]


def load_eval_cameras(reference_model_dir):
    cfg.model_path = reference_model_dir
    cfg.trained_model_dir = os.path.join(cfg.model_path, "trained_model")
    cfg.point_cloud_dir = os.path.join(cfg.model_path, "point_cloud")

    original_mode = cfg.mode
    cfg.mode = "train"
    scene_info = readWaymoFullInfo(cfg.source_path, **cfg.data)

    cameras = []
    if ARGS.split in ["train", "all"]:
        cameras.extend(cameraList_from_camInfos(scene_info.train_cameras, 1))
    if ARGS.split in ["test", "all"]:
        cameras.extend(cameraList_from_camInfos(scene_info.test_cameras, 1))

    cameras = sorted(cameras, key=lambda camera: camera.id)
    if ARGS.max_views > 0:
        cameras = cameras[: ARGS.max_views]

    cfg.mode = original_mode
    return scene_info.metadata, cameras


def checkpoint_path(model_dir):
    trained_model_dir = os.path.join(model_dir, "trained_model")
    iteration = ARGS.iteration
    if iteration < 0:
        iteration = searchForMaxIteration(trained_model_dir)
    path = os.path.join(trained_model_dir, f"iteration_{iteration}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path, iteration


def load_model(model_dir, metadata):
    cfg.model_path = model_dir
    cfg.trained_model_dir = os.path.join(model_dir, "trained_model")
    cfg.point_cloud_dir = os.path.join(model_dir, "point_cloud")
    cfg.mode = "evaluate"

    ckpt_path, iteration = checkpoint_path(model_dir)
    gaussians = StreetGaussianModel(metadata)
    state_dict = torch.load(ckpt_path, map_location="cuda")
    gaussians.load_state_dict(state_dict)
    gaussians.eval()
    return gaussians, iteration


def get_valid_mask(camera, result, gt_inv_depth, gt_depth):
    valid = gt_inv_depth > 0
    valid = valid & torch.isfinite(gt_depth) & (gt_depth > 0) & (gt_depth < ARGS.max_depth)

    if ARGS.mask == "ground":
        if "gt_ground_mask" not in camera.guidance:
            raise KeyError("gt_ground_mask is missing; rerun with --mask valid.")
        valid = valid & camera.guidance["gt_ground_mask"].squeeze().cuda(non_blocking=True).bool()

    if ARGS.min_acc > 0:
        acc = result["acc"].squeeze(0)
        valid = valid & (acc > ARGS.min_acc)

    return valid


def update_metric(stats, name, values, mask, channels=1):
    count = mask.sum().item()
    if count == 0:
        return
    stats[f"{name}_sum"] += values[mask].sum().item()
    stats[f"{name}_count"] += count * channels


@torch.no_grad()
def evaluate_model(model_dir, metadata, cameras):
    gaussians, iteration = load_model(model_dir, metadata)
    renderer = StreetGaussianRenderer()
    stats = defaultdict(float)
    per_view = {}

    desc = os.path.basename(model_dir)
    for camera in tqdm(cameras, desc=f"Evaluating {desc}"):
        if "gt_depth" not in camera.guidance:
            raise KeyError(
                "gt_depth is missing. This script needs data/waymo/training/002/gt_depth."
            )

        result = renderer.render(camera, gaussians)

        gt_inv_depth = camera.guidance["gt_depth"].squeeze().cuda(non_blocking=True).float()
        gt_depth = 1.0 / (gt_inv_depth + 1e-9)
        gt_depth = torch.clip(gt_depth, 0.0, ARGS.max_depth)

        pred_depth = result["depth"].squeeze(0)
        if not ARGS.raw_depth:
            pred_depth = pred_depth / (result["acc"].squeeze(0) + 1e-10)
        pred_depth = torch.clip(pred_depth, 0.0, ARGS.max_depth)

        valid = get_valid_mask(camera, result, gt_inv_depth, gt_depth)
        depth_abs = torch.abs(pred_depth - gt_depth)
        update_metric(stats, "depth_mae", depth_abs, valid)

        pred_normals = result["normals"]
        gt_normals = renderer.render_normal(camera, gt_depth)
        pred_norm = torch.linalg.norm(pred_normals, dim=0)
        gt_norm = torch.linalg.norm(gt_normals, dim=0)
        normal_valid = valid & (pred_norm > 1e-6) & (gt_norm > 1e-6)

        pred_normals = F.normalize(pred_normals, p=2, dim=0)
        gt_normals = F.normalize(gt_normals, p=2, dim=0)
        normal_abs = torch.abs(pred_normals - gt_normals).sum(dim=0)
        update_metric(stats, "normal_l1_mae", normal_abs, normal_valid, channels=3)

        cos_error = 1.0 - torch.sum(pred_normals * gt_normals, dim=0).clamp(-1.0, 1.0)
        update_metric(stats, "normal_cos_mae", cos_error, normal_valid)

        angle_error = torch.rad2deg(torch.acos((1.0 - cos_error).clamp(-1.0, 1.0)))
        update_metric(stats, "normal_angle_mae_deg", angle_error, normal_valid)

        stats["views"] += 1
        stats["valid_depth_pixels"] += valid.sum().item()
        stats["valid_normal_pixels"] += normal_valid.sum().item()

        if ARGS.per_view:
            name = camera.image_name
            per_view[name] = {
                "depth_mae": masked_mean(depth_abs, valid),
                "normal_l1_mae": masked_mean(normal_abs / 3.0, normal_valid),
                "normal_cos_mae": masked_mean(cos_error, normal_valid),
                "normal_angle_mae_deg": masked_mean(angle_error, normal_valid),
                "valid_depth_pixels": int(valid.sum().item()),
                "valid_normal_pixels": int(normal_valid.sum().item()),
            }

    del gaussians
    torch.cuda.empty_cache()

    summary = {
        "iteration": iteration,
        "views": int(stats["views"]),
        "valid_depth_pixels": int(stats["valid_depth_pixels"]),
        "valid_normal_pixels": int(stats["valid_normal_pixels"]),
        "depth_mae": divide(stats["depth_mae_sum"], stats["depth_mae_count"]),
        "normal_l1_mae": divide(stats["normal_l1_mae_sum"], stats["normal_l1_mae_count"]),
        "normal_cos_mae": divide(stats["normal_cos_mae_sum"], stats["normal_cos_mae_count"]),
        "normal_angle_mae_deg": divide(
            stats["normal_angle_mae_deg_sum"],
            stats["normal_angle_mae_deg_count"],
        ),
    }
    return summary, per_view


def divide(num, den):
    return float("nan") if den == 0 else num / den


def masked_mean(values, mask):
    count = mask.sum().item()
    if count == 0:
        return float("nan")
    return values[mask].mean().item()


def print_table(results):
    headers = [
        "model",
        "depth_mae",
        "normal_l1_mae",
        "normal_cos_mae",
        "normal_angle_mae_deg",
        "views",
    ]
    widths = [28, 12, 16, 16, 22, 8]
    print(" ".join(header.rjust(width) for header, width in zip(headers, widths)))
    print(" ".join("-" * width for width in widths))
    for model_name, item in results.items():
        row = [
            model_name,
            f"{item['depth_mae']:.6f}",
            f"{item['normal_l1_mae']:.6f}",
            f"{item['normal_cos_mae']:.6f}",
            f"{item['normal_angle_mae_deg']:.6f}",
            str(item["views"]),
        ]
        print(" ".join(value.rjust(width) for value, width in zip(row, widths)))


def main():
    model_dirs = resolve_model_dirs()
    missing = [path for path in model_dirs if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError("Missing model dirs:\n" + "\n".join(missing))

    cfg.render.render_normal = True
    cfg.eval.quiet = True

    metadata, cameras = load_eval_cameras(model_dirs[0])
    if len(cameras) == 0:
        raise RuntimeError(f"No cameras found for split={ARGS.split}")

    results = {}
    per_view_results = {}
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        summary, per_view = evaluate_model(model_dir, metadata, cameras)
        results[model_name] = summary
        if ARGS.per_view:
            per_view_results[model_name] = per_view

    print_table(results)

    output_json = ARGS.output_json
    if not output_json:
        output_json = os.path.join(
            os.path.dirname(model_dirs[0]),
            f"{ARGS.exp_prefix}_normal_depth_eval_{ARGS.split}.json",
        )

    payload = {
        "config": ARGS.config,
        "split": ARGS.split,
        "mask": ARGS.mask,
        "raw_depth": ARGS.raw_depth,
        "max_depth": ARGS.max_depth,
        "min_acc": ARGS.min_acc,
        "model_dirs": model_dirs,
        "results": results,
    }
    if ARGS.per_view:
        payload["per_view"] = per_view_results

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to {output_json}")


if __name__ == "__main__":
    main()
