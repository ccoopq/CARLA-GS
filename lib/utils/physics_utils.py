import numpy as np
from physics_integration import StreetGSPhysicsIntegrator
from tqdm import tqdm
import cv2
import os
from lib.config import cfg

class PhysicsRenderer:
    """
    Enhanced renderer that handles physics simulation for colliding objects.
    """
    def __init__(self, physics_config_path="PhysGaussian/config/hit.json"):
        self.physics_integrator = None
        self.physics_integrator = StreetGSPhysicsIntegrator(physics_config_path)
        
        self.collision_detected = False
        self.collision_frame = None
        self.physics_pose_mappings = {}
        self.simulation_active = False
        
    def handle_collision_and_simulate(self, cameras, gaussians, collision_info, simulation_frames):
        if collision_info is None:
            return
            
        self.collision_detected = True
        self.collision_frame = collision_info['frame_idx']
        
        print(f"\n=== PHYSICS SIMULATION ===") 
        print(f"Running physics simulation for {simulation_frames} frames after collision at frame {self.collision_frame}")
        
        # Run physics simulation
        result = self.physics_integrator.handle_collision(cameras, gaussians, collision_info)
        self.simulation_active = True
        print(f"Physics simulation completed.")
        return result
        
            
    def render_frame_with_physics(self, cameras, gaussians, renderer, visualizer, physics_result=None):
        rgbs = []
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            # Render the frame
            result = renderer.render_all(camera, gaussians)
            rgb = (result['rgb'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgbs.append(rgb)
        rgbs.extend(physics_result) # add physics results to visualizer frames
        
        # save imgs
        output_path = 'PhysGaussian/output/hit_031'
        assert output_path is not None
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for frame, rgb in enumerate(tqdm(rgbs)):
            height = rgb.shape[0] // 2 * 2
            width = rgb.shape[1] // 2 * 2
            cv2.imwrite(os.path.join(output_path, f"{frame}.png".rjust(8, "0")), rgb)

        # save video
        if True:
            fps = cfg.render.fps
            os.system(
                f"ffmpeg -framerate {fps} -i {output_path}/%04d.png -c:v libx264 -s {width}x{height} -y -pix_fmt yuv420p {output_path}/output.mp4"
            )

def find_all_collisions(cameras, gaussians, thresh, obj_list=None):
    """
    Find collisions between all pairs of objects in the scene.
    
    Args:
        cameras: List of cameras
        gaussians: The gaussian model
        obj_list: List of object IDs to check for collisions (if None, check all)
        thresh: Distance threshold for collision detection
    
    Returns:
        Dictionary with collision information, or None if no collisions found
    """
    print("Checking collisions between all objects")
    collision_results = []
    
    if obj_list is None:
        # Get all object IDs
        obj_ids = [name for name in gaussians.model_name_id.keys() if name.startswith('obj_')]
        print(f"Found objects: {obj_ids}")
    
        if len(obj_ids) < 2:
            print("Need at least 2 objects for collision detection")
            return None

        # Check all pairs of objects
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                id1, id2 = obj_ids[i], obj_ids[j]
                print(f"\nChecking collision between {id1} and {id2}")
                
                collision_info = find_collision_frame_pair(cameras, gaussians, id1, id2, thresh)
                if collision_info is not None:
                    collision_results.append(collision_info)
    else:
        # Check specified object pairs
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                id1, id2 = obj_list[i], obj_list[j]
                print(f"\nChecking collision between {id1} and {id2}")
                
                collision_info = find_collision_frame_pair(cameras, gaussians, id1, id2, thresh)
                if collision_info is not None:
                    collision_results.append(collision_info)
    
    if collision_results:
        # Sort by frame index to find earliest collision
        collision_results.sort(key=lambda x: x['frame_idx'])
        earliest = collision_results[0]
        
        print(f"\n=== COLLISION SUMMARY ===")
        print(f"Total collisions found: {len(collision_results)}")
        print(f"Earliest collision: {earliest['obj1']} vs {earliest['obj2']} at frame {earliest['frame_idx']}")
        for result in collision_results:
            print(f"  Frame {result['frame_idx']}: {result['obj1']} vs {result['obj2']} (distance={result['distance']:.3f}, pos1={result['pos1']}, pos2={result['pos2']}, speed1={result['speed1']}, speed2={result['speed2']})")

        return earliest
    else:
        print("\nNo collisions found between any objects")
        return None


def calculate_object_speed(cameras, gaussians, obj_id, current_idx, lookback_frames=15):
    """
    Calculate the estimated speed of an object using polynomial prediction based on its position trajectory.
    
    Args:
        cameras: List of cameras
        gaussians: The gaussian model
        obj_id: Object ID (e.g., 'obj_000')
        current_idx: Current frame index in the cameras list
        lookback_frames: Number of previous frames to use for speed calculation
    
    Returns:
        Speed magnitude in units per frame using polynomial derivative, or 0.0 if insufficient data
    """
    if current_idx < lookback_frames:
        return 0.0
    
    positions = []
    frame_indices = []
    
    # Collect positions from previous frames
    for i in range(max(0, current_idx - lookback_frames), current_idx):
        camera = cameras[i]
        
        # Set visibility and parse camera
        gaussians.set_visibility([obj_id])
        gaussians.parse_camera(camera)
        
        # Check if object is visible in this frame
        if obj_id not in gaussians.graph_obj_list:
            continue
            
        # Get object center position
        bbox = get_object_bbox(gaussians, obj_id, camera)
        if bbox is not None:
            positions.append(bbox['center'])
            frame_indices.append(camera.meta['frame_idx'])
    
    if len(positions) < 3:  # Need at least 3 points for polynomial fitting
        return 0.0
    
    # Calculate speed using polynomial prediction
    positions = np.array(positions)
    frame_indices = np.array(frame_indices)
    
    try:
        # Fit polynomial to each coordinate (x, y, z) separately
        speeds = []
        
        # Create visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Object {obj_id} Trajectory Analysis (Frame {current_idx})', fontsize=16)
        
        coord_names = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        
        for coord in range(3):  # x, y, z coordinates
            coord_positions = positions[:, coord]
            
            # Fit polynomial (degree 2 for smooth trajectory, or degree 1 for linear)
            degree = min(2, len(positions) - 1)  # Use degree 2 if we have enough points
            poly_coeffs = np.polyfit(frame_indices, coord_positions, degree)
            
            # Calculate derivative (velocity) at the most recent frame
            poly_derivative = np.polyder(poly_coeffs)
            current_frame = frame_indices[-2] # Use the second last frame for velocity calculation!!!
            velocity_coord = np.polyval(poly_derivative, current_frame)
            speeds.append(velocity_coord)
            
            # Generate smooth curve for visualization
            frame_range = np.linspace(frame_indices[0], frame_indices[-1], 100)
            fitted_curve = np.polyval(poly_coeffs, frame_range)
            
            # Plot on subplot
            ax = axes[coord]
            
            # Plot original points
            ax.scatter(frame_indices, coord_positions, color=colors[coord], s=50, 
                      label=f'Observed {coord_names[coord]} positions', alpha=0.7, zorder=3)
            
            # Plot fitted curve
            ax.plot(frame_range, fitted_curve, color=colors[coord], linewidth=2, 
                   label=f'Polynomial fit (degree {degree})', zorder=2)
            
            # Highlight current frame
            current_pos = np.polyval(poly_coeffs, current_frame)
            ax.scatter([current_frame], [current_pos], color='black', s=100, 
                      marker='*', label=f'Current frame', zorder=4)
            
            # Add velocity arrow (show direction)
            arrow_length = velocity_coord * 2  # Scale for visibility
            ax.arrow(current_frame, current_pos, 2, arrow_length, 
                    head_width=0.5, head_length=0.3, fc='orange', ec='orange',
                    label=f'Velocity: {velocity_coord:.3f}')
            
            # Formatting
            ax.set_xlabel('Frame Index')
            ax.set_ylabel(f'{coord_names[coord]} Position')
            ax.set_title(f'{coord_names[coord]}-coordinate\nVelocity: {velocity_coord:.3f} units/frame')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add text with polynomial equation
            poly_text = f'P(t) = '
            for i, coeff in enumerate(poly_coeffs):
                power = len(poly_coeffs) - 1 - i
                if power == 0:
                    poly_text += f'{coeff:.3f}'
                elif power == 1:
                    poly_text += f'{coeff:.3f}t + '
                else:
                    poly_text += f'{coeff:.3f}t^{power} + '
            ax.text(0.05, 0.95, poly_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
                   fontsize=8, verticalalignment='top')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = 'PhysGaussian/output/trajectory_analysis'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_filename = f'trajectory_{obj_id}_frame_{current_idx:04d}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"Trajectory analysis plot saved: {plot_path}")
        
        velocity_vector = np.array(speeds)
        return velocity_vector
        
    except (np.linalg.LinAlgError, np.RankWarning):
        return None


def find_collision_frame_pair(cameras, gaussians, id1, id2, thresh=4.5):
    """
    Find the frame where two specific objects collide based on bounding box intersection.
    
    Args:
        cameras: List of cameras
        gaussians: The gaussian model
        id1: First object ID (e.g., 'obj_000')
        id2: Second object ID (e.g., 'obj_001')
        thresh: Distance threshold for collision detection
    
    Returns:
        Dictionary with collision info, or None if no collision found
    """
    # Get track IDs from object names
    track_id1 = int(id1.split('_')[1])
    track_id2 = int(id2.split('_')[1])
    
    # Check if both objects exist
    if track_id1 not in gaussians.obj_info or track_id2 not in gaussians.obj_info:
        print(f"One or both objects not found: {id1}, {id2}")
        return None
    
    min_distance = float('inf')
    
    for idx, camera in enumerate(cameras):
        # Set visibility and parse camera
        gaussians.set_visibility([id1, id2])
        gaussians.parse_camera(camera)
        
        # Check if both objects are visible in this frame
        if id1 not in gaussians.graph_obj_list or id2 not in gaussians.graph_obj_list:
            continue
            
        # Get bounding boxes for both objects separately
        bbox1 = get_object_bbox(gaussians, id1, camera)
        bbox2 = get_object_bbox(gaussians, id2, camera)
        
        if bbox1 is None or bbox2 is None:
            continue
            
        # Check for bounding box intersection
        intersection_volume = compute_bbox_intersection(bbox1, bbox2)
        
        # Calculate center distance
        center1 = (bbox1['min'] + bbox1['max']) / 2
        center2 = (bbox2['min'] + bbox2['max']) / 2
        distance = np.linalg.norm(center1 - center2)
        
        if distance < min_distance:
            min_distance = distance
        
        # Check for collision (intersection volume > 0 or distance < threshold)
        if intersection_volume > 0 or distance < thresh:
            # Calculate speeds for both objects
            speed1 = calculate_object_speed(cameras, gaussians, id1, idx)
            speed2 = calculate_object_speed(cameras, gaussians, id2, idx)

            # update box, center to the previous frame!!!
            if idx > 0:
                bbox1 = get_object_bbox(gaussians, id1, cameras[idx-1])
                bbox2 = get_object_bbox(gaussians, id2, cameras[idx-1])
                center1 = (bbox1['min'] + bbox1['max']) / 2
                center2 = (bbox2['min'] + bbox2['max']) / 2

            collision_info = {
                'obj1': id1,
                'obj2': id2,
                'frame_idx': camera.meta['frame_idx'],
                'distance': distance,
                'intersection_volume': intersection_volume,
                'bbox1': bbox1,
                'bbox2': bbox2,
                'pos1': center1,
                'pos2': center2,
                'speed1': speed1,
                'speed2': speed2
            }
            print(f"Collision detected: {id1} vs {id2} at frame {camera.meta['frame_idx']}: distance={distance:.3f}, intersection_volume={intersection_volume:.6f}, speed1={speed1}, speed2={speed2}")
            return collision_info
            
        if idx % 20 == 0:  # Print progress every 20 frames
            print(f"  Frame {camera.meta['frame_idx']}: distance={distance:.3f}")
    
    print(f"No collision found between {id1} and {id2}. Minimum distance was {min_distance:.3f}")
    return None


def find_collision_frame(cameras, gaussians, id1, id2, thresh=0.05):
    """
    Legacy function for backward compatibility.
    Find the frame where two objects collide based on bounding box intersection.
    """
    result = find_collision_frame_pair(cameras, gaussians, id1, id2, thresh)
    return result['frame_idx'] if result else None


def get_object_bbox(gaussians, obj_id, camera):
    """
    Get the bounding box of an object.
    
    Args:
        gaussians: The gaussian model 
        obj_id: Object ID (e.g., 'obj_000')
        camera: Camera to parse for this object
    
    Returns:
        Dictionary with 'min' and 'max' coordinates, or None if object not found
    """
    # Set visibility to only this object and parse camera
    gaussians.set_visibility([obj_id])
    gaussians.parse_camera(camera)
    
    # Get object points
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    if xyz.shape[0] == 0:
        return None
        
    # Compute bounding box
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    
    return {
        'min': bbox_min,
        'max': bbox_max,
        'size': bbox_max - bbox_min,
        'center': (bbox_min + bbox_max) / 2
    }


def compute_bbox_intersection(bbox1, bbox2):
    """
    Compute the intersection volume of two bounding boxes.
    
    Args:
        bbox1: First bounding box dict with 'min' and 'max'
        bbox2: Second bounding box dict with 'min' and 'max'
    
    Returns:
        Intersection volume (0 if no intersection)
    """
    # Compute intersection bounds
    intersection_min = np.maximum(bbox1['min'], bbox2['min'])
    intersection_max = np.minimum(bbox1['max'], bbox2['max'])
    
    # Check if intersection exists
    if np.any(intersection_min >= intersection_max):
        return 0.0
    
    # Compute intersection volume
    intersection_size = intersection_max - intersection_min
    volume = np.prod(intersection_size)
    
    return volume
