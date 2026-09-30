import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import numpy as np
from lib.utils.general_utils import quaternion_raw_multiply, create_rotation_quaternion

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
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
                
def render_trajectory():
    cfg.render.save_image = True
    cfg.render.save_video = False
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        # A.  Remove objects from the scene
        # print("obj_info keys before:", gaussians.obj_info.keys())
        # track_id_000 = getattr(gaussians, 'obj_000').track_id
        # del gaussians.obj_info[track_id_000]
        # gaussians.obj_list.remove('obj_000')
        # del gaussians.model_name_id['obj_000']
        # delattr(gaussians, 'obj_000')
        # print("obj_info keys after:", gaussians.obj_info.keys())

        # B. Rotate an object in the scene
        # obj_id = 'obj_004'
        # angle_degrees = -15.0 
        # axis = 'z'  # Choose 'x', 'y', or 'z'
        # if hasattr(gaussians, 'actor_pose'):
        #     track_id = int(obj_id.split('_')[1])
        #     angle_rad = np.radians(angle_degrees)
        #     dq = create_rotation_quaternion(angle_degrees, axis=axis)

        #     track_idx = gaussians.obj_info[track_id]['track_idx']
        #     frame_idx, column_idx = track_idx[:, 0], track_idx[:, 1]
        #     for i in range(len(frame_idx)):
        #         f, c = frame_idx[i], column_idx[i]
        #         current_rot = gaussians.actor_pose.input_rots[f, c]
        #         new_rot = quaternion_raw_multiply(current_rot, dq)
        #         gaussians.actor_pose.input_rots[f, c] = new_rot

        #     if gaussians.actor_pose.opt_track:
        #         gaussians.actor_pose.opt_rots[frame_idx, column_idx] = torch.zeros_like(gaussians.actor_pose.opt_rots[frame_idx, column_idx])
        # else:
        #     raise RuntimeError("Scene does not support object pose manipulation")
        
        # C. Translate an object in the scene
        # obj_id = 'obj_000'
        # translation_vector = torch.tensor([0.0, -4.0, 0.0], dtype=torch.float32, device='cuda')
        # if hasattr(gaussians, 'actor_pose'):
        #     track_id = int(obj_id.split('_')[1])
        #     track_idx = gaussians.obj_info[track_id]['track_idx']
        #     frame_idx, column_idx = track_idx[:, 0], track_idx[:, 1]
        #     for i in range(len(frame_idx)):
        #         f, c = frame_idx[i], column_idx[i]
        #         current_trans = gaussians.actor_pose.input_trans[f, c]
        #         new_trans = current_trans + translation_vector
        #         gaussians.actor_pose.input_trans[f, c] = new_trans

        #     if gaussians.actor_pose.opt_track:
        #         gaussians.actor_pose.opt_trans[frame_idx, column_idx] = torch.zeros_like(gaussians.actor_pose.opt_trans[frame_idx, column_idx])
        # else:
        #     raise RuntimeError("Scene does not support object translation manipulation")

        scene = Scene(gaussians=gaussians, dataset=dataset)
        
        # D. Copy objects from the scene (done AFTER scene loading to avoid loading issues)
        # source_obj_name = 'obj_004'
        # target_obj_name = 'obj_999'
        
        # if hasattr(gaussians, source_obj_name):
        #     # Get the source object
        #     source_obj: GaussianModelActor = getattr(gaussians, source_obj_name)
            
        #     # Create a copy of the object metadata
        #     source_track_id = source_obj.track_id
        #     if source_track_id in gaussians.obj_info:
        #         # Create new track_id for the copied object
        #         new_track_id = 999  # Choose an unused track_id
                
        #         # Copy object metadata
        #         obj_meta_copy = gaussians.obj_info[source_track_id].copy()
        #         obj_meta_copy['track_id'] = new_track_id
                
        #         # Create new object instance
        #         new_obj = GaussianModelActor(model_name=target_obj_name, obj_meta=obj_meta_copy)
                
        #         # Copy the Gaussian parameters from source object
        #         # This copies all the 3D Gaussian properties (position, rotation, scale, color, etc.)
        #         new_obj._xyz = source_obj._xyz.clone()
        #         new_obj._features_dc = source_obj._features_dc.clone()
        #         new_obj._features_rest = source_obj._features_rest.clone()
        #         new_obj._scaling = source_obj._scaling.clone()
        #         new_obj._rotation = source_obj._rotation.clone()
        #         new_obj._opacity = source_obj._opacity.clone()
                
        #         # Apply translation offset to differentiate the copy
        #         translation_offset = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
        #         new_obj._xyz = new_obj._xyz + translation_offset
                
        #         # Add the new object to the gaussians model
        #         setattr(gaussians, target_obj_name, new_obj)
        #         gaussians.obj_list.append(target_obj_name)
        #         gaussians.model_name_id[target_obj_name] = gaussians.models_num
        #         gaussians.models_num += 1
        #         gaussians.obj_info[new_track_id] = obj_meta_copy
                
        #         print(f"Successfully copied {source_obj_name} to {target_obj_name} after scene loading")
        #         print(f"obj_info keys after copying:", gaussians.obj_info.keys())
        #     else:
        #         print(f"Track ID {source_track_id} not found in obj_info")
        # else:
        #     print(f"Source object {source_obj_name} not found")
        
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

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
