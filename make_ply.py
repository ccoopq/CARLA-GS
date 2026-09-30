import numpy as np
import os
import shutil
from lib.config import cfg
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model import GaussianModel
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from plyfile import PlyData, PlyElement


inverse_opacity = lambda x: np.log(x/(1-x))
inverse_scale = lambda x: np.log(x)

if __name__ == '__main__':
    frame_id = cfg.viewer.frame_id 
    
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    cameras = train_cameras + test_cameras
    cameras = list(sorted(cameras, key=lambda x: x.id))
    
    viewpoint_camera = None
    for camera in cameras:
        if camera.meta['frame_idx'] == frame_id:
            viewpoint_camera = camera
            break
        
    if viewpoint_camera is None:
        raise ValueError(f'Could not find camera with frame_idx {frame_id}')
    
    save_dir = os.path.join(cfg.model_path, 'viewer', f'{frame_id:06d}')
    pointcloud_dir = os.path.join(save_dir, 'point_cloud', f'iteration_{cfg.train.iterations}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(pointcloud_dir, exist_ok=True)
    shutil.copyfile(os.path.join(cfg.model_path, 'cameras.json'), os.path.join(save_dir, 'cameras.json'))
    shutil.copyfile(os.path.join(cfg.model_path, 'cfg_args'), os.path.join(save_dir, 'cfg_args'))
    shutil.copyfile(os.path.join(cfg.model_path, 'input.ply'), os.path.join(save_dir, 'input.ply'))

    # Save each object's point cloud independently
    for model_name in gaussians.model_name_id.keys():
        print(f"Saving point cloud for object: {model_name}")
        
        # Set visibility to only the current model
        gaussians.set_visibility([model_name])
        gaussians.parse_camera(camera=viewpoint_camera)
        
        # Get the object-specific Gaussian parameters
        xyz = gaussians.get_xyz.detach().cpu().numpy()    
        if xyz.shape[0] == 0:
            print(f"Skipping {model_name} - no points found")
            continue
        normals = np.zeros_like(xyz)
        
        f = gaussians.get_features.detach().transpose(1, 2).contiguous() # [n, 3, sh_degree]
        f_dc = f[..., :1].flatten(start_dim=1).cpu().numpy()
        f_rest = f[..., 1:].flatten(start_dim=1).cpu().numpy()
        opacities = gaussians.get_opacity.detach().cpu().numpy()
        opacities = np.clip(opacities, a_min=1e-6, a_max=1.-1e-6)
        opacities = inverse_opacity(opacities)

        print(f"  Points: {xyz.shape[0]}, Features: {f_dc.shape}, Rest: {f_rest.shape}, Opacity: {opacities.shape}")
        
        scale = gaussians.get_scaling.detach().cpu().numpy()
        scale = inverse_scale(scale)
        
        rotation = gaussians.get_rotation.detach().cpu().numpy()

        # Skip if this object has no points
        if xyz.shape[0] == 0:
            print(f"Skipping {model_name} - no points found")
            continue

        # Center the object point cloud at its centroid
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid

        # Build the PLY structure
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        # Save individual object point cloud
        elements = PlyElement.describe(elements, 'vertex')
        object_ply_path = os.path.join(pointcloud_dir, f'point_cloud_{model_name}.ply')
        PlyData([elements]).write(object_ply_path)
        print(f"Saved {model_name} point cloud to: {object_ply_path}")

    # Also save the combined point cloud (all objects together)
    print("Saving combined point cloud for all objects")
    gaussians.set_visibility(list(set(gaussians.model_name_id.keys())))
    gaussians.parse_camera(camera=viewpoint_camera)
    
    xyz = gaussians.get_xyz.detach().cpu().numpy()    
    normals = np.zeros_like(xyz)
    
    f = gaussians.get_features.detach().transpose(1, 2).contiguous() # [n, 3, sh_degree]
    f_dc = f[..., :1].flatten(start_dim=1).cpu().numpy()
    f_rest = f[..., 1:].flatten(start_dim=1).cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy()
    opacities = np.clip(opacities, a_min=1e-6, a_max=1.-1e-6)
    opacities = inverse_opacity(opacities)
    
    scale = gaussians.get_scaling.detach().cpu().numpy()
    scale = inverse_scale(scale)
    
    rotation = gaussians.get_rotation.detach().cpu().numpy()

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(f_rest.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    elements = PlyElement.describe(elements, 'vertex')
    combined_ply_path = os.path.join(pointcloud_dir, 'point_cloud_combined.ply')
    PlyData([elements]).write(combined_ply_path)
    print(f"Saved combined point cloud to: {combined_ply_path}")

    # save semantic=1 to point cloud
    if cfg.data.use_semantic:
        print("Saving point cloud with semantic == 1")
        semantics = gaussians.get_semantic.detach().cpu().numpy()

        # argmax
        sem_labels = np.argmax(semantics, axis=1)
        print(f"Semantic shape: {semantics.shape}")
        print(f"Semantic labels shape: {sem_labels.shape}")
        mask = sem_labels == 1

        # softmax
        # exp_sem = np.exp(semantics - np.max(semantics, axis=1, keepdims=True))  # 防止溢出
        # sem_softmax = exp_sem / np.sum(exp_sem, axis=1, keepdims=True)
        # print(sem_softmax[:50])
        # mask = sem_softmax[:, 1] > 0.995

        if mask.sum() == 0:
            print("No points with semantic == 1 found.")
        else:
            xyz_sem = xyz[mask]
            normals_sem = normals[mask]
            f_dc_sem = f_dc[mask]
            f_rest_sem = f_rest[mask]
            opacities_sem = opacities[mask]
            scale_sem = scale[mask]
            rotation_sem = rotation[mask]

            l_sem = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            for i in range(f_dc_sem.shape[1]):
                l_sem.append(f'f_dc_{i}')
            for i in range(f_rest_sem.shape[1]):
                l_sem.append(f'f_rest_{i}')
            l_sem.append('opacity')
            for i in range(scale_sem.shape[1]):
                l_sem.append(f'scale_{i}')
            for i in range(rotation_sem.shape[1]):
                l_sem.append(f'rot_{i}')
            dtype_sem = [(attribute, 'f4') for attribute in l_sem]

            elements_sem = np.empty(xyz_sem.shape[0], dtype=dtype_sem)
            attributes_sem = np.concatenate(
                (xyz_sem, normals_sem, f_dc_sem, f_rest_sem, opacities_sem, scale_sem, rotation_sem), axis=1
            )
            elements_sem[:] = list(map(tuple, attributes_sem))
            elements_sem = PlyElement.describe(elements_sem, 'vertex')
            semantic_ply_path = os.path.join(pointcloud_dir, 'point_cloud_sem.ply')
            PlyData([elements_sem]).write(semantic_ply_path)
            print(f"Saved semantic point cloud to: {semantic_ply_path}")

        