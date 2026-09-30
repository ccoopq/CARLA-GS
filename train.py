import os
import cv2
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim, lpips, get_img_grad_weight, compute_flat_loss, patch_based_ncc_loss, angular_loss, geo_consist_loss, geo_consist_loss_3x3
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state, get_expon_lr_func
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
import numpy as np
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training():
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data

    start_iter = 0
    tb_writer = prepare_output_and_logger()
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)

    gaussians.training_setup()
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')
        state_dict = torch.load(ckpt_path)
        start_iter = state_dict['iter']
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)
    except:
        pass

    print(f'Starting from {start_iter}')
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    gaussians_renderer = StreetGaussianRenderer()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_ssim_for_log = 0.0
    ema_lpips_for_log = 0.0
    # psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        gt_image = viewpoint_cam.original_image
        mask = viewpoint_cam.guidance['mask'] if 'mask' in viewpoint_cam.guidance else torch.ones_like(gt_image[0:1]).bool()
        gt_image = gt_image.cuda(non_blocking=True) if not gt_image.is_cuda else gt_image
        mask = mask.cuda(non_blocking=True) if not mask.is_cuda else mask
        if 'lidar_depth' in viewpoint_cam.guidance:
            lidar_depth = viewpoint_cam.guidance['lidar_depth']
            lidar_depth = lidar_depth.cuda(non_blocking=True) if not lidar_depth.is_cuda else lidar_depth
        if 'sky_mask' in viewpoint_cam.guidance:
            sky_mask = viewpoint_cam.guidance['sky_mask']
            sky_mask = sky_mask.cuda(non_blocking=True) if not sky_mask.is_cuda else sky_mask
        if 'obj_bound' in viewpoint_cam.guidance:
            obj_bound = viewpoint_cam.guidance['obj_bound']
            obj_bound = obj_bound.cuda(non_blocking=True) if not obj_bound.is_cuda else obj_bound
        if 'gt_depth' in viewpoint_cam.guidance:
            gt_depth = viewpoint_cam.guidance['gt_depth']
            gt_depth = gt_depth.cuda(non_blocking=True) if not gt_depth.is_cuda else gt_depth
        if 'gt_ground_mask' in viewpoint_cam.guidance:
            gt_ground_mask = viewpoint_cam.guidance['gt_ground_mask']
            gt_ground_mask = gt_ground_mask.cuda(non_blocking=True) if not gt_ground_mask.is_cuda else gt_ground_mask
        if 'gt_semantics' in viewpoint_cam.guidance:
            gt_semantics = viewpoint_cam.guidance['gt_semantics']
            gt_semantics = gt_semantics.cuda(non_blocking=True) if not gt_semantics.is_cuda else gt_semantics

        # render all
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image, acc, viewspace_point_tensor, visibility_filter, radii = render_pkg["rgb"], render_pkg['acc'], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth = render_pkg['depth'] # [1, H, W]
        normals = render_pkg['normals'] # [3, H, W]
        semantics = render_pkg['semantics'] # [num_classes, H, W]
        
        # render_obj
        render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
            # back to all scenes
        include_list = list(set(gaussians.model_name_id.keys()))
        gaussians.set_visibility(include_list)


        scalar_dict = dict()
        
        # rgb loss
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image, gt_image, mask=mask))
    
        # sky loss
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        # obj reg loss
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss

            # back to all scenes
            include_list = list(set(gaussians.model_name_id.keys()))
            gaussians.set_visibility(include_list)

        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:            
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            expected_depth = depth / (render_pkg['acc'] + 1e-10)  
            depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
            depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
            lidar_depth_loss = depth_error.mean()
            scalar_dict['lidar_depth_loss'] = lidar_depth_loss.item()
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss

        # flat loss
        if visibility_filter.sum() > 0:
            if gaussians.get_scaling.shape[0] != visibility_filter.shape[0]:
                continue
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            scalar_dict['flat_loss'] = min_scale_loss.mean().item()
            loss += optim_args.lambda_flat * min_scale_loss.mean()

        # depth + normal + geo loss
        gt_depth = gt_depth.squeeze(0)
        gt_inv_depth = (gt_depth.to("cuda")).float()
        gt_depth = 1.0 / (gt_inv_depth + 1e-9)
        gt_depth = torch.clip(gt_depth, 0.0, 1000.0)
        depth = torch.clip(depth, 0.0, 1000.0)
        depth = depth.squeeze(0)
        inv_depth = 1.0 / (depth + 1e-9)
        inf_depth_mask = gt_inv_depth!=0

        acc_obj = render_pkg_obj['acc']
        obj_mask = acc_obj.squeeze() > 0.8
        obj_ground_mask = (obj_mask | gt_ground_mask) & inf_depth_mask

        gt_normal = gaussians_renderer.render_normal(viewpoint_cam, gt_depth)

        if iteration > 0 and gt_depth is not None:         
            # depth_loss = patch_based_ncc_loss(inv_depth, gt_inv_depth, obj_ground_mask)
            normal_loss = angular_loss(normals, gt_normal, obj_ground_mask)
            geo_consistency_loss = geo_consist_loss_3x3(normals, gt_depth, obj_ground_mask)

            # scalar_dict['depth_loss'] = depth_loss
            scalar_dict['normal_loss'] = normal_loss.item()
            scalar_dict['geo_consistency_loss'] = geo_consistency_loss.item()

            loss += optim_args.lambda_normal * normal_loss
            loss += optim_args.lambda_geo * geo_consistency_loss
        
        # semantics loss only for cam0
        cam_name = viewpoint_cam.meta.get('cam', '')
        if cam_name == 0 and iteration > 0 and optim_args.lambda_semantic > 0:
            gt = (gt_semantics > 0.5).long().squeeze(0).clamp(0, 1)
            assert semantics.shape[1:] == gt.shape, f"Shape mismatch: {semantics.shape} vs {gt.shape}"
            semantic_loss = F.cross_entropy(semantics.unsqueeze(0), gt.unsqueeze(0))
            scalar_dict['semantic_loss'] = semantic_loss.item()
            # loss += optim_args.lambda_semantic * semantic_loss

        
        scalar_dict['loss'] = loss.item()

        loss.backward()

        iter_end.record()
                
        is_save_images = True
        if is_save_images and (iteration % 500 == 0):
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
                    # back to all scenes
                include_list = list(set(gaussians.model_name_id.keys()))
                gaussians.set_visibility(include_list)

            depth_colored, _ = visualize_depth_numpy(render_pkg["depth"].detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)

            rend_normal = render_pkg['normals'].permute(1,2,0)
            rend_normal = rend_normal/(rend_normal.norm(dim=-1, keepdim=True)+1.0e-8)
            rend_normal = rend_normal.detach().cpu().numpy()
            rend_normal = ((rend_normal+1) * 127.5).astype(np.uint8).clip(0, 255)
            rend_normal = torch.from_numpy(rend_normal).permute(2, 0, 1).float().cuda() / 255.0 

            gt_normal = gt_normal.permute(1,2,0)
            gt_normal = gt_normal/(gt_normal.norm(dim=-1, keepdim=True)+1.0e-8)
            gt_normal = gt_normal.detach().cpu().numpy()
            gt_normal = ((gt_normal+1) * 127.5).astype(np.uint8).clip(0, 255)   
            gt_normal = torch.from_numpy(gt_normal).permute(2, 0, 1).float().cuda() / 255.0

            obj_ground_mask = obj_ground_mask.repeat(3, 1, 1)

            row1 = torch.cat([obj_ground_mask, rend_normal*obj_ground_mask, gt_normal*obj_ground_mask], dim=2)

            acc = acc.repeat(3, 1, 1)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = acc_obj.repeat(3, 1, 1)
            row2 = torch.cat([acc, image_obj, acc_obj], dim=2)

            # rend_semantics = render_pkg['semantics']
            # sem_pred = torch.argmax(rend_semantics, dim=0).detach().cpu().numpy()
            # num_classes = rend_semantics.shape[0]
            # colors = np.array([
            #     [0, 0, 0],      
            #     [255, 255, 255]  
            # ], dtype=np.uint8)
            # sem_pred = colors[sem_pred]  # [H, W, 3]
            # sem_pred = torch.from_numpy(sem_pred).permute(2, 0, 1).float().cuda() / 255.0  # [3, H, W]
            # gt_semantics = gt_semantics.repeat(3, 1, 1)
            # row3 = torch.cat([gt_semantics, gt_semantics, sem_pred], dim=2)

            image_to_show = torch.cat([row0, row1, row2], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
        with torch.no_grad():
            
            # Log
            tensor_dict = dict()

            if iteration % 10 == 0:                    
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
                ema_ssim_for_log = 0.4 * ssim(image, gt_image, mask=mask).mean().float() + 0.6 * ema_ssim_for_log
                ema_lpips_for_log = 0.4 * lpips(image, gt_image, mask=mask).mean().float() + 0.6 * ema_lpips_for_log
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{4}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}",
                                          "SSIM": f"{ema_ssim_for_log:.{4}f}",
                                          "LPIPS": f"{ema_lpips_for_log:.{4}f}"}, refresh=False)
            progress_bar.update(1)
            # if iteration == training_args.iterations:
            #     progress_bar.close()

            # Save ply
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))
                gaussians.set_max_radii2D(radii, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                prune_big_points = iteration > optim_args.opacity_reset_interval

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )

                        scalar_dict.update(scalars)
                        tensor_dict.update(tensors)
                        
            # Reset opacity
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene, gaussians_renderer)

            # Optimizer step
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                torch.save(state_dict, ckpt_path)



def prepare_output_and_logger():
    
    # if cfg.model_path == '':
    #     if os.getenv('OAR_JOB_ID'):
    #         unique_str = os.getenv('OAR_JOB_ID')
    #     else:
    #         unique_str = str(uuid.uuid4())
    #     cfg.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(cfg.model_path))

    os.makedirs(cfg.model_path, exist_ok=True)
    os.makedirs(cfg.trained_model_dir, exist_ok=True)
    os.makedirs(cfg.record_dir, exist_ok=True)
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))

    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree
        viewer_arg['white_background'] = cfg.data.white_background
        viewer_arg['source_path'] = cfg.source_path
        viewer_arg['model_path']= cfg.model_path
        cfg_log_f.write(str(Namespace(**viewer_arg)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, renderer: StreetGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()
                    ssim_test += ssim(image, gt_image, mask=mask).mean().double()
                    lpips_test += lpips(image, gt_image, mask=mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)

    # Initialize system state (RNG)
    safe_state(cfg.train.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)
    training()

    # All done
    print("\nTraining complete.")