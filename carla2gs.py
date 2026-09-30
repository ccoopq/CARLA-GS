import torch 
import os
import json
import torch.nn as nn
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.datasets.dataset import Dataset
from lib.datasets.base_readers import fetchPly
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.utils.general_utils import inverse_sigmoid
from lib.utils.sh_utils import RGB2SH, SH2RGB
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import numpy as np
from lib.utils.general_utils import quaternion_raw_multiply, create_rotation_quaternion
from simple_knn._C import distCUDA2
from plyfile import PlyData
import csv

# Fill this dict to replace objects before rendering.
# Example:
# REPLACEMENT_MODELS = {
#     "12": {
#         "ply_path": "/absolute/path/to/new_car.ply",
#         "manual_yaw_deg": None,         # set a float to force yaw if auto-rotation is wrong
#         "scale_mode": "isotropic",     # "isotropic" or "anisotropic"
#         "center_mode": "bbox",         # "bbox" or "mean"
#     }
# }
REPLACEMENT_MODELS = {}  # keyed by track id, e.g. {"84": {"ply_path": "./SAM3D_MODELS/031_object_084.ply", ...}}


def _rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _compute_center(xyz: np.ndarray, mode: str = 'bbox') -> np.ndarray:
    if mode == 'mean':
        return xyz.mean(axis=0)
    if mode == 'bbox':
        return (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
    raise ValueError(f'Unsupported center mode: {mode}')


def _yaw_from_min_area_bbox(xy: np.ndarray) -> float:
    if xy.shape[0] < 3:
        return 0.0

    best_yaw = 0.0
    best_area = float('inf')

    # Use only bounding-box geometry: yaw with minimum XY bbox area.
    for yaw in np.linspace(0.0, np.pi, 721, endpoint=False):
        c, s = np.cos(yaw), np.sin(yaw)
        rot2 = np.array([[c, -s], [s, c]], dtype=np.float32)
        rotated_xy = xy @ rot2.T
        ext = np.maximum(np.ptp(rotated_xy, axis=0), 1e-6)
        area = float(ext[0] * ext[1])
        if area < best_area:
            best_area = area
            best_yaw = float(yaw)

    return best_yaw


def _safe_extent(xyz: np.ndarray) -> np.ndarray:
    return np.maximum(np.ptp(xyz, axis=0), 1e-6)


def _sample_points(points: np.ndarray, max_points: int = 2000) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def _chamfer_like_distance(a: np.ndarray, b: np.ndarray, max_points: int = 2000) -> float:
    a_s = _sample_points(a, max_points=max_points)
    b_s = _sample_points(b, max_points=max_points)

    # Pairwise squared distances [Na, Nb]
    diff = a_s[:, None, :] - b_s[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)

    a_to_b = np.min(dist2, axis=1).mean()
    b_to_a = np.min(dist2, axis=0).mean()
    return float(a_to_b + b_to_a)


def _compute_scale(orig_extent: np.ndarray, new_extent: np.ndarray, scale_mode: str):
    ratio = orig_extent / np.maximum(new_extent, 1e-6)
    if scale_mode == 'anisotropic':
        return ratio.astype(np.float32)
    if scale_mode == 'isotropic':
        return np.float32(np.median(ratio))
    raise ValueError(f'Unsupported scale mode: {scale_mode}')


def _align_replacement_points(
    original_xyz: np.ndarray,
    replacement_xyz: np.ndarray,
    manual_yaw_deg: float = None,
    scale_mode: str = 'isotropic',
    center_mode: str = 'bbox',
):
    if original_xyz.shape[0] == 0 or replacement_xyz.shape[0] == 0:
        raise RuntimeError('Original/replacement ply has no points.')

    original_center = _compute_center(original_xyz, mode=center_mode)
    replacement_center = _compute_center(replacement_xyz, mode=center_mode)

    original_centered = original_xyz - original_center
    replacement_centered = replacement_xyz - replacement_center

    if manual_yaw_deg is None:
        yaw_original = _yaw_from_min_area_bbox(original_centered[:, :2])
        yaw_replacement = _yaw_from_min_area_bbox(replacement_centered[:, :2])
        yaw_delta = yaw_original - yaw_replacement
        yaw_candidates = [yaw_delta, yaw_delta + np.pi, yaw_delta + np.pi/2, yaw_delta - np.pi/2]
    else:
        yaw_candidates = [np.deg2rad(manual_yaw_deg)]

    original_extent = _safe_extent(original_centered)

    best_error = float('inf')
    best_result = None
    original_for_score = original_centered.astype(np.float32)

    for yaw in yaw_candidates:
        rot = _rotation_matrix_z(yaw)
        rotated = replacement_centered @ rot.T
        replacement_extent = _safe_extent(rotated)
        # Enforce bbox ratio matching in x/y/z using anisotropic scale.
        scale = (original_extent / np.maximum(replacement_extent, 1e-6)).astype(np.float32)
        scaled = rotated * scale
        extent_error = np.linalg.norm(_safe_extent(scaled) - original_extent)
        shape_error = _chamfer_like_distance(original_for_score, scaled.astype(np.float32))
        total_error = extent_error + shape_error

        if total_error < best_error:
            best_error = total_error
            best_result = {
                'yaw_rad': yaw,
                'scale': scale,
                'scaled': scaled,
                'extent_error': extent_error,
                'shape_error': shape_error,
            }

    aligned_xyz = best_result['scaled'] + original_center
    return aligned_xyz, {
        'yaw_deg': float(np.degrees(best_result['yaw_rad'])),
        'yaw_rad': float(best_result['yaw_rad']),
        'scale': best_result['scale'],
        'extent_error': float(best_result['extent_error']),
        'shape_error': float(best_result['shape_error']),
        'original_center': original_center,
        'replacement_center': replacement_center,
    }


def _sorted_prop_names(ply_vertex, prefix: str):
    names = [p.name for p in ply_vertex.properties if p.name.startswith(prefix)]
    return sorted(names, key=lambda x: int(x.split('_')[-1]))


def _resize_feature_dim(t: torch.Tensor, target_dim: int) -> torch.Tensor:
    # t: [N, D, 3]
    if t.shape[1] == target_dim:
        return t
    if t.shape[1] > target_dim:
        return t[:, :target_dim, :]
    pad = torch.zeros((t.shape[0], target_dim - t.shape[1], t.shape[2]), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


def _load_replacement_gaussian_params(replacement_ply, obj_model: GaussianModelActor):
    xyz = np.vstack([
        np.asarray(replacement_ply['x']),
        np.asarray(replacement_ply['y']),
        np.asarray(replacement_ply['z']),
    ]).T.astype(np.float32)

    n = xyz.shape[0]
    replacement_fields = set(replacement_ply.data.dtype.names or [])

    # DC features
    f_dc_names = _sorted_prop_names(replacement_ply, 'f_dc_')
    if len(f_dc_names) > 0:
        f_dc = np.stack([np.asarray(replacement_ply[nm]) for nm in f_dc_names], axis=1).astype(np.float32)
    else:
        if all(k in replacement_fields for k in ('red', 'green', 'blue')):
            rgb = np.vstack([
                np.asarray(replacement_ply['red']),
                np.asarray(replacement_ply['green']),
                np.asarray(replacement_ply['blue']),
            ]).T.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
        elif all(k in replacement_fields for k in ('r', 'g', 'b')):
            rgb = np.vstack([
                np.asarray(replacement_ply['r']),
                np.asarray(replacement_ply['g']),
                np.asarray(replacement_ply['b']),
            ]).T.astype(np.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
        else:
            rgb = np.ones((n, 3), dtype=np.float32)
        f_dc = RGB2SH(torch.tensor(rgb, dtype=torch.float32)).cpu().numpy()

    f_dc = f_dc.reshape(n, 3, -1)
    f_dc = torch.tensor(f_dc, dtype=torch.float32, device='cuda').transpose(1, 2).contiguous()  # [N, D, 3]
    f_dc = _resize_feature_dim(f_dc, obj_model.fourier_dim)

    # SH rest features
    target_rest_dim = (obj_model.max_sh_degree + 1) ** 2 - 1
    f_rest_names = _sorted_prop_names(replacement_ply, 'f_rest_')
    if len(f_rest_names) > 0:
        f_rest = np.stack([np.asarray(replacement_ply[nm]) for nm in f_rest_names], axis=1).astype(np.float32)
        f_rest = f_rest.reshape(n, 3, -1)
        f_rest = torch.tensor(f_rest, dtype=torch.float32, device='cuda').transpose(1, 2).contiguous()
    else:
        f_rest = torch.zeros((n, 0, 3), dtype=torch.float32, device='cuda')
    f_rest = _resize_feature_dim(f_rest, target_rest_dim)

    # Opacity
    if 'opacity' in replacement_fields:
        opacity = np.asarray(replacement_ply['opacity']).astype(np.float32).reshape(n, 1)
        opacity = torch.tensor(opacity, dtype=torch.float32, device='cuda')
    else:
        opacity = inverse_sigmoid(0.1 * torch.ones((n, 1), dtype=torch.float32, device='cuda'))

    # Scaling (raw/log-domain)
    scale_names = _sorted_prop_names(replacement_ply, 'scale_')
    if len(scale_names) > 0:
        scaling = np.stack([np.asarray(replacement_ply[nm]) for nm in scale_names], axis=1).astype(np.float32)
        if scaling.shape[1] >= 3:
            scaling = scaling[:, :3]
        else:
            pad = np.zeros((n, 3 - scaling.shape[1]), dtype=np.float32)
            scaling = np.concatenate([scaling, pad], axis=1)
        scaling = torch.tensor(scaling, dtype=torch.float32, device='cuda')
    else:
        if n > 1:
            dist2 = torch.clamp_min(distCUDA2(torch.tensor(xyz, dtype=torch.float32, device='cuda')), 1e-7)
            scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        else:
            scaling = torch.zeros((n, 3), dtype=torch.float32, device='cuda')

    # Rotation (raw quaternion)
    rot_names = _sorted_prop_names(replacement_ply, 'rot_')
    if len(rot_names) > 0:
        rotation = np.stack([np.asarray(replacement_ply[nm]) for nm in rot_names], axis=1).astype(np.float32)
        if rotation.shape[1] >= 4:
            rotation = rotation[:, :4]
        else:
            pad = np.zeros((n, 4 - rotation.shape[1]), dtype=np.float32)
            rotation = np.concatenate([rotation, pad], axis=1)
        rotation = torch.tensor(rotation, dtype=torch.float32, device='cuda')
    else:
        rotation = torch.zeros((n, 4), dtype=torch.float32, device='cuda')
        rotation[:, 0] = 1.0

    # Semantic
    semantic_names = _sorted_prop_names(replacement_ply, 'semantic_')
    if len(semantic_names) > 0 and obj_model.num_classes > 0:
        semantic = np.stack([np.asarray(replacement_ply[nm]) for nm in semantic_names], axis=1).astype(np.float32)
        semantic = torch.tensor(semantic, dtype=torch.float32, device='cuda')
        if semantic.shape[1] > obj_model.num_classes:
            semantic = semantic[:, :obj_model.num_classes]
        elif semantic.shape[1] < obj_model.num_classes:
            pad = torch.zeros((n, obj_model.num_classes - semantic.shape[1]), dtype=torch.float32, device='cuda')
            semantic = torch.cat([semantic, pad], dim=1)
    else:
        semantic = torch.zeros((n, obj_model.num_classes), dtype=torch.float32, device='cuda')

    return {
        'xyz': xyz,
        'features_dc': f_dc,
        'features_rest': f_rest,
        'opacity': opacity,
        'scaling': scaling,
        'rotation': rotation,
        'semantic': semantic,
    }


def _apply_alignment_to_gaussian_params(params: dict, aligned_xyz: np.ndarray, yaw_rad: float, scale_xyz: np.ndarray):
    n = aligned_xyz.shape[0]
    params['xyz'] = torch.tensor(aligned_xyz, dtype=torch.float32, device='cuda')

    # Compose local rotations with global yaw rotation.
    half = 0.5 * float(yaw_rad)
    q_global = torch.tensor([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=torch.float32, device='cuda')
    q_global = q_global.unsqueeze(0).expand(n, -1)
    params['rotation'] = quaternion_raw_multiply(q_global, params['rotation'])

    # Keep replacement Gaussian scales, then apply global bbox scale in log-domain.
    s = torch.tensor(np.maximum(scale_xyz, 1e-6), dtype=torch.float32, device='cuda')
    params['scaling'] = params['scaling'] + torch.log(s).unsqueeze(0)


def _replace_actor_model_from_gaussian_params(obj_model: GaussianModelActor, params: dict):
    obj_model._xyz = nn.Parameter(params['xyz'].requires_grad_(True))
    obj_model._features_dc = nn.Parameter(params['features_dc'].requires_grad_(True))
    obj_model._features_rest = nn.Parameter(params['features_rest'].requires_grad_(True))
    obj_model._scaling = nn.Parameter(params['scaling'].requires_grad_(True))
    obj_model._rotation = nn.Parameter(params['rotation'].requires_grad_(True))
    obj_model._opacity = nn.Parameter(params['opacity'].requires_grad_(True))
    obj_model._semantic = nn.Parameter(params['semantic'].requires_grad_(True))
    obj_model.max_radii2D = torch.zeros((obj_model.get_xyz.shape[0]), device='cuda')


def align_and_replace_object_model_runtime(
    gaussians: StreetGaussianModel,
    track_id,
    replacement_ply_path: str,
    manual_yaw_deg: float = None,
    scale_mode: str = 'isotropic',
    center_mode: str = 'bbox',
):
    track_name = f'obj_{str(track_id).zfill(3)}'

    if not hasattr(gaussians, track_name):
        raise ValueError(f'Object {track_name} not found in current Gaussian scene.')
    if not os.path.exists(replacement_ply_path):
        raise FileNotFoundError(f'Replacement ply not found: {replacement_ply_path}')

    obj_model: GaussianModelActor = getattr(gaussians, track_name)
    original_xyz = obj_model.get_xyz.detach().cpu().numpy().astype(np.float32)

    replacement_ply = PlyData.read(replacement_ply_path)['vertex']
    replacement_params = _load_replacement_gaussian_params(replacement_ply, obj_model)
    replacement_xyz = replacement_params['xyz']

    aligned_xyz, meta = _align_replacement_points(
        original_xyz=original_xyz,
        replacement_xyz=replacement_xyz,
        manual_yaw_deg=manual_yaw_deg,
        scale_mode=scale_mode,
        center_mode=center_mode,
    )

    _apply_alignment_to_gaussian_params(
        params=replacement_params,
        aligned_xyz=aligned_xyz,
        yaw_rad=meta['yaw_rad'],
        scale_xyz=meta['scale'],
    )
    _replace_actor_model_from_gaussian_params(obj_model, replacement_params)
    print(f'[PLY ALIGN] runtime replaced {track_name} from {replacement_ply_path}')
    print(f'[PLY ALIGN] yaw(deg): {meta["yaw_deg"]:.3f}, scale: {meta["scale"]}')
    print(f'[PLY ALIGN] extent_error: {meta["extent_error"]:.6f}, shape_error: {meta["shape_error"]:.6f}')
    print(f'[PLY ALIGN] center original: {meta["original_center"]}, replacement: {meta["replacement_center"]}')


def prepare_replacement_models(gaussians: StreetGaussianModel):
    if len(REPLACEMENT_MODELS) == 0:
        return

    print('[PLY ALIGN] preparing runtime replacement models ...')
    for track_id, setting in REPLACEMENT_MODELS.items():
        align_and_replace_object_model_runtime(
            gaussians=gaussians,
            track_id=track_id,
            replacement_ply_path=setting['ply_path'],
            manual_yaw_deg=setting.get('manual_yaw_deg', None),
            scale_mode=setting.get('scale_mode', 'isotropic'),
            center_mode=setting.get('center_mode', 'bbox'),
        )

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        prepare_replacement_models(gaussians)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))

def load_carla_trajectory():
    carla_traj = []
    csv_path = cfg.carla_traj
    if not csv_path:
        raise ValueError('Set the CARLA trajectory CSV with: carla_traj path/to/target_vehicle_trajectory.csv')
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            timestep = float(row['frame_id'])
            x = float(row['x'])
            y = float(row['y'])
            heading = float(row['heading'])
            track_id = str(row['track_id'])
            
            traj_point = {
                "t": timestep,
                "x": x,
                "y": y,
                "heading": heading,
                "track_id": track_id
            }
            carla_traj.append(traj_point)
    return carla_traj
                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        # replace 3D models
        prepare_replacement_models(gaussians)

        # replace with the carla trajectory
        carla_traj = load_carla_trajectory()
        for traj in carla_traj:
            # frame_idx = int(traj['t'] * 10)  # assuming 10 FPS
            frame_idx = int(traj['t'])
            track_id = traj['track_id']
            track_name = f"obj_{track_id.zfill(3)}"
            for camera in cameras:
                # if camera.meta['frame_idx'] + cfg.data.selected_frames[0] == frame_idx: # offset
                if camera.meta['frame_idx'] == frame_idx: # no offset
                    for model_name in gaussians.model_name_id.keys():
                        if model_name == track_name:
                            # gaussians.set_visibility([model_name])
                            include_list = list(set(gaussians.model_name_id.keys()))
                            gaussians.set_visibility(include_list)
                            gaussians.parse_camera(camera)
                            if len(gaussians.graph_obj_list) > 0:
                                if hasattr(gaussians, 'actor_pose'):
                                    print(f"Setting object {model_name} for frame {frame_idx}")
                                    print(f"current objs in the scene: {gaussians.graph_obj_list}")
                                    track_idx = gaussians.obj_info[int(track_id)]['track_idx']
                                    track_frame_idx = track_idx[:, 0]
                                    track_column_idx = track_idx[:, 1]
                                    frame_match = track_frame_idx == camera.meta['frame_idx']
                                    if not torch.any(frame_match):
                                        print(f"No track slot for object {model_name} at frame {camera.meta['frame_idx']}, skip update")
                                        continue
                                    column_idx = track_column_idx[frame_match][0]
                                    # trans
                                    current_trans = gaussians.actor_pose.input_trans[camera.meta['frame_idx'], column_idx]
                                    translation_vector = torch.tensor([traj['x'], traj['y'], 0.0], dtype=torch.float32, device='cuda')
                                    translation_vector[2] = current_trans[2]  # keep original z
                                    print(f"Original position: {current_trans}, rotation: {gaussians.actor_pose.input_rots[camera.meta['frame_idx'], column_idx]}")
                                    gaussians.actor_pose.input_trans[camera.meta['frame_idx'], column_idx] = translation_vector
                                    #rots
                                    # gaussians.actor_pose.input_rots[camera.meta['frame_idx'], column_idx] = create_rotation_quaternion(-traj['heading'] + np.pi/2, axis='z') # 002
                                    gaussians.actor_pose.input_rots[camera.meta['frame_idx'], column_idx] = create_rotation_quaternion(2*np.pi + np.degrees(traj['heading']), axis='z') # 031
                                    print(f"Updated position: {translation_vector}, rotation: {gaussians.actor_pose.input_rots[camera.meta['frame_idx'], column_idx]}")
                            break
                    break
        
        # render
        start_time = time.time()
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)
        end_time = time.time()
        print(f"Total rendering time: {end_time - start_time:.2f} seconds")
        print(f"Total rendering rate: {len(cameras) / (end_time - start_time):.2f} frames/second")

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    else:
        raise NotImplementedError()
    

