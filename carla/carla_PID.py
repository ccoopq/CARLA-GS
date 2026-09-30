import random
import math
import time
import sys
import os

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from agents.navigation.controller import VehiclePIDController

import matplotlib.pyplot as plt
from collections import deque
import csv
import argparse


def normalize_angle_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_ego_frame(target_transform, ego_transform):
    dx = target_transform.location.x - ego_transform.location.x
    dy = target_transform.location.y - ego_transform.location.y

    ego_yaw = math.radians(ego_transform.rotation.yaw)
    target_yaw = math.radians(target_transform.rotation.yaw)

    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    x_rel = cos_yaw * dx + sin_yaw * dy
    y_rel = -sin_yaw * dx + cos_yaw * dy
    heading_rel = normalize_angle_rad(target_yaw - ego_yaw)

    return x_rel, - y_rel, - heading_rel


# ============================================================
# Load trajectories from CSV
# ============================================================
def load_trajectory_from_file(csv_path, m):
    target_traj = []
    ego_traj = []
    
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        
        for row in reader:
            timestep = float(row['timestep'])
            frame_id = int(row['frame_id'])
            x = float(row['x'])
            y = float(row['y'])
            heading = float(row['heading'])
            trajectory_type = str(row['trajectory_type'])
            track_id = str(row['vehicle_id'])
            
            location = carla.Location(x=x, y=-y, z=0.5)  # flip y to match the CARLA coordinate frame
            rotation =  carla.Rotation(yaw=math.degrees( - heading))  # Convert radians to degrees
            transform = carla.Transform(location, rotation)
            
            # Create a simple waypoint-like object
            waypoint = type('obj', (object,), {
                'transform': transform
            })
            
            traj_point = {
                "t": timestep,
                "frame_id": frame_id,
                "waypoint": waypoint,
                "type": trajectory_type,
                "track_id": track_id
            }
            
            # Split based on trajectory type
            if 'target' in trajectory_type:
                target_traj.append(traj_point)
            elif 'ego' in trajectory_type:
                ego_traj.append(traj_point)
    
    print(f"Loaded {len(target_traj)} target points and {len(ego_traj)} ego points")
    return target_traj, ego_traj


# ============================================================
# Spawn random static vehicles (obstacles)
# ============================================================
def spawn_obstacles(world, m, road_id, road_length, lane_ids, count=5):
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter("vehicle.*")
    spawned = []

    for _ in range(count):
        s = random.uniform(20, road_length - 20)
        lane_id = random.choice(lane_ids)

        wp = m.get_waypoint_xodr(road_id=road_id, lane_id=lane_id, s=s)
        if wp is None:
            continue

        transform = wp.transform
        transform.location.z += 0.1

        if any(v.get_location().distance(transform.location) < 8.0 for v in spawned):
            continue

        bp = random.choice(vehicle_bps)
        bp.set_attribute("role_name", "obstacle")
        vehicle = world.try_spawn_actor(bp, transform)

        if vehicle:
            vehicle.set_autopilot(False)
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            spawned.append(vehicle)

    return spawned


# ============================================================
# Build a time-stamped reference trajectory (the LLM output format)
# ============================================================
def generate_random_behavior_trajectory(
    m,
    road_id,
    lane_ids,
    s_start,
    s_end,
    base_ds=2.0,
    base_dt=0.2,
    lane_change_prob=0.15,
    lane_change_cooldown_steps=10,
    accel_noise_std=0.05,
    seed=None
):
    """
    Time-parameterized reference trajectory with:
    - ONLY adjacent lane changes
    - lane-change cooldown (no frequent weaving)
    - random acceleration / deceleration via dt perturbation
    """

    if seed is not None:
        random.seed(seed)

    ref_traj = []

    # initial state
    s = s_start
    t = 0.0
    cur_lane = random.choice(lane_ids)

    ds = base_ds
    dt = base_dt

    lane_cooldown = 0  # steps remaining where lane change is forbidden

    while s <= s_end:
        wp = m.get_waypoint_xodr(
            road_id=road_id,
            lane_id=cur_lane,
            s=s
        )

        if wp is None:
            s += ds
            t += dt
            continue

        wp.transform.location.z += 0.1

        ref_traj.append({
            "t": t,
            "waypoint": wp,
            "lane_id": cur_lane
        })

        # ======================================
        # 1. Lane change logic (STRICT version)
        # ======================================
        if lane_cooldown > 0:
            lane_cooldown -= 1
        else:
            if random.random() < lane_change_prob:
                # only adjacent lanes allowed
                adjacent_lanes = []
                if cur_lane - 1 in lane_ids:
                    adjacent_lanes.append(cur_lane - 1)
                if cur_lane + 1 in lane_ids:
                    adjacent_lanes.append(cur_lane + 1)

                if adjacent_lanes:
                    cur_lane = random.choice(adjacent_lanes)
                    lane_cooldown = lane_change_cooldown_steps

        # ======================================
        # 2. Random acceleration / deceleration
        # ======================================
        dt_noise = random.gauss(0.0, accel_noise_std)
        dt = max(0.05, base_dt + dt_noise)

        # ======================================
        # 3. Advance longitudinal position & time
        # ======================================
        s += ds
        t += dt

    return ref_traj


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Track LLM waypoints with a PID controller in CARLA.")
    parser.add_argument("--input", required=True, help="LLM corner-case trajectory CSV (from LLMCorner/main.py)")
    parser.add_argument("--output", default="target_vehicle_trajectory.csv", help="CARLA-executed trajectory CSV")
    parser.add_argument("--xodr", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "super_road.xodr"),
                        help="OpenDRIVE map used for execution")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    xodr_path = args.xodr
    with open(xodr_path, 'r') as f:
        xodr_data = f.read()

    world = client.generate_opendrive_world(xodr_data)
    world.tick()

    m = world.get_map()

    ROAD_ID = 20
    ROAD_LENGTH = 200

    # ----------------------
    # Load and split trajectories
    # ----------------------
    target_traj, ego_traj = load_trajectory_from_file(args.input, m)

    # ----------------------
    # Spawn target vehicle (PID controlled)
    # ----------------------
    if len(target_traj) > 0:
        start_transform = target_traj[0]["waypoint"].transform
        start_transform.location.z += 0.1
        start_transform.rotation.yaw = 180.0 # -x
    else:
        print("❌ No target trajectory points loaded")
        return

    bp_target = world.get_blueprint_library().find("vehicle.audi.tt")
    bp_target.set_attribute("role_name", "hero")
    target_vehicle = world.try_spawn_actor(bp_target, start_transform)

    if target_vehicle is None:
        print("❌ Target vehicle spawn failed")
        return

    # ----------------------
    # Spawn ego vehicle (position controlled)
    # ----------------------
    ego_vehicle = None
    if len(ego_traj) > 0:
        ego_start_transform = ego_traj[0]["waypoint"].transform
        ego_start_transform.location.z += 0.1
        
        bp_ego = world.get_blueprint_library().find("vehicle.tesla.model3")
        bp_ego.set_attribute("role_name", "ego")
        ego_vehicle = world.try_spawn_actor(bp_ego, ego_start_transform)
        
        if ego_vehicle is None:
            print("❌ Ego vehicle spawn failed")

    # set initial speed for target vehicle
    v0 = 0.0  # m/s
    yaw = target_vehicle.get_transform().rotation.yaw
    heading = math.radians(yaw)
    velocity = carla.Vector3D(
        v0 * math.cos(heading),
        v0 * math.sin(heading),
        0.0
    )
    target_vehicle.set_target_velocity(velocity)

    # Draw trajectory points
    for i in range(len(target_traj)):
        world.debug.draw_point(
            target_traj[i]["waypoint"].transform.location,
            size=0.1,
            color=carla.Color(255, 0, 0),  # Red for target
            life_time=60.0
        )
    
    for i in range(len(ego_traj)):
        world.debug.draw_point(
            ego_traj[i]["waypoint"].transform.location,
            size=0.1,
            color=carla.Color(0, 255, 0),  # Green for ego
            life_time=60.0
        )

    # ----------------------
    # Set up camera to follow target vehicle
    # ----------------------
    spectator = world.get_spectator()
    
    # ----------------------
    # PID controller for target vehicle
    # ----------------------
    pid = VehiclePIDController(
        target_vehicle,
        args_lateral={'K_P': 2.0, 'K_D': 0.2, 'K_I': 0.0},
        args_longitudinal={'K_P': 2.0, 'K_D': 0.2, 'K_I': 0.0},
        max_throttle=1.0,
        max_brake=1.0,
        max_steering=1.0
    )

    # =====================================================
    # Real-time visualization (for PID tuning)
    # =====================================================
    plt.ion()  # interactive mode

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ax[0].set_ylabel("X position (m)")
    ax[1].set_ylabel("Y position (m)")
    ax[2].set_ylabel("Position error (m)")
    ax[2].set_xlabel("Time (s)")

    for a in ax:
        a.grid(True)

    # Use fixed-length buffers to avoid memory blow-up
    max_len = 300
    t_buf = deque(maxlen=max_len)
    x_ref_buf = deque(maxlen=max_len)
    y_ref_buf = deque(maxlen=max_len)
    x_cur_buf = deque(maxlen=max_len)
    y_cur_buf = deque(maxlen=max_len)
    err_buf = deque(maxlen=max_len)

    # Plot handles
    line_x_ref, = ax[0].plot([], [], 'r--', label="x_ref")
    line_x_cur, = ax[0].plot([], [], 'b-',  label="x")

    line_y_ref, = ax[1].plot([], [], 'r--', label="y_ref")
    line_y_cur, = ax[1].plot([], [], 'b-',  label="y")

    line_err,   = ax[2].plot([], [], 'k-', label="||pos error||")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()


    # ----------------------
    # Time-aware tracking
    # ----------------------
    start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
    dt = 0.2  # not follow the real time exactly, set larger for precise PID control
    
    # ----------------------
    # CSV output setup
    # ----------------------
    output_data = [None] * len(target_traj)  # Pre-allocate array with same length
    
    print("\n🚗 Time-parameterized trajectory tracking started\n")




    while True:
        world.tick()
        snap = world.get_snapshot()
        t_now = snap.timestamp.elapsed_seconds - start_sim_time

        # ========= 1. Update target vehicle with PID =========
        target_idx = int(t_now / dt)
        if target_idx >= len(target_traj):
            print("\n🎯 Target trajectory completed")
            break

        target_wp = target_traj[target_idx]["waypoint"]
        
        # Calculate target vehicle control
        cur_loc = target_vehicle.get_location()
        ref_loc = target_wp.transform.location
        dist = cur_loc.distance(ref_loc)

        if target_idx > 0:
            prev_wp = target_traj[target_idx - 1]["waypoint"]
            ds = prev_wp.transform.location.distance(target_wp.transform.location)
            v_ref = ds / dt
        else:
            v_ref = 0.0

        v_cmd = v_ref + 10 * dist

        # Apply PID control to target vehicle
        control = pid.run_step(target_speed=v_cmd, waypoint=target_wp)
        target_vehicle.apply_control(control)

        # ========= 2. Update ego vehicle position directly =========
        if ego_vehicle and target_idx < len(ego_traj):
            ego_transform = ego_traj[target_idx]["waypoint"].transform
            print("!!!!!", ego_transform)
            ego_vehicle.set_transform(ego_transform)

        # ========= 3. Record target vehicle data at reference timesteps =========
        target_transform = target_vehicle.get_transform()

        # Per-frame ego reference transform
        if ego_vehicle and target_idx < len(ego_traj):
            ego_transform = ego_vehicle.get_transform()
        elif target_idx < len(ego_traj):
            ego_transform = ego_traj[target_idx]["waypoint"].transform
        else:
            ego_transform = carla.Transform(
                target_transform.location,
                target_transform.rotation
            )

        ego_x_rel, ego_y_rel, ego_heading_rel = world_to_ego_frame(
            target_transform,
            ego_transform
        )
        
        # Use the exact timestep from reference trajectory
        ref_timestep = target_traj[target_idx]["t"]
        
        output_data[target_idx] = {
            'timestep': ref_timestep,
            'x': ego_x_rel,
            'y': ego_y_rel,
            'heading': ego_heading_rel,
            'track_id': target_traj[target_idx]["track_id"]
        }

        # ========= 4. Debug output =========
        sys.stdout.write(
            f"\rt={t_now:5.2f}s | idx={target_idx:03d} "
            f"| dist={dist:5.2f} m "
            f"| v_cmd={v_cmd:6.2f} m/s "
            f"| pos=({cur_loc.x:6.1f}, {cur_loc.y:6.1f}) "
        )
        sys.stdout.flush()

        # =====================================================
        # Update camera to follow target vehicle
        # =====================================================
        vehicle_transform = ego_vehicle.get_transform()
        camera_location = vehicle_transform.location + carla.Location(z=5, x=-10)
        camera_rotation = carla.Rotation(pitch=-10, yaw=vehicle_transform.rotation.yaw)
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))

        # =====================================================
        # Update real-time plot
        # =====================================================
        cur_loc = target_vehicle.get_location()
        ref_loc = target_wp.transform.location

        pos_err = cur_loc.distance(ref_loc)

        t_buf.append(t_now)
        x_ref_buf.append(ref_loc.x)
        y_ref_buf.append(ref_loc.y)
        x_cur_buf.append(cur_loc.x)
        y_cur_buf.append(cur_loc.y)
        err_buf.append(pos_err)

        line_x_ref.set_data(t_buf, x_ref_buf)
        line_x_cur.set_data(t_buf, x_cur_buf)

        line_y_ref.set_data(t_buf, y_ref_buf)
        line_y_cur.set_data(t_buf, y_cur_buf)

        line_err.set_data(t_buf, err_buf)

        for a in ax:
            a.relim()
            a.autoscale_view()

        plt.pause(0.001)

    # ----------------------
    # Save target vehicle trajectory to CSV
    # ----------------------
    output_file = args.output
    with open(output_file, 'w', newline='') as file:
        fieldnames = ['frame_id', 'timestep', 'x', 'y', 'heading', 'track_id']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        # Filter out None values (in case simulation ended early)
        valid_data = [data for data in output_data if data is not None]
        for frame_count, row in enumerate(valid_data):
            row_with_frame = dict(row)
            row_with_frame['frame_id'] = frame_count + target_traj[0]["frame_id"]  # Start frame_id from the first target trajectory point
            writer.writerow(row_with_frame)
    
    print(f"\n✅ Target vehicle trajectory saved to {output_file}")
    print(f"Reference trajectory length: {len(target_traj)}")
    print(f"Recorded trajectory length: {len(valid_data)}")
    print("Coordinates: ego_x=front, ego_y=left (ego-centered per timestep)")


if __name__ == "__main__":
    main()
