"""
Improved Safety-Critical Scenario Generation Framework
Innovation: Target-Oriented Approach with Collision Zones

Key Innovation:
- Replace acceleration-based generation with collision zone targeting
- Enable "creating scenarios from scratch" rather than just perturbation
- LLM directly selects collision zones for maximum threat
"""

import math
import os
import pandas as pd
import numpy as np
import json
import ast
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ImprovedSafetyCriticalGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4", fps: int = 10):
        """Initialize the improved generator with OpenAI API"""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fps = fps
        
        # Behavior library
        self.behavior_library = [
            "Sudden braking of the lead vehicle",
            "Sudden cut-in from adjacent lane",
            "Rear tailgating in the same lane",
            "Side pass (close lateral overtake)",
            "Sudden appearance from blind spot (unocclusion)",
            "Sudden start from parked position in the front as least 20m (ghost jump)",
            "Opposing vehicle drifting over lane (near head-on collision)",
            "Lead vehicle swerving largely to avoid obstacle (S-shape path)"
        ]
        
        # Ego vehicle dimensions (can be customized)
        self.ego_length = 4.5  # meters
        self.ego_width = 2.0   # meters
        
    def load_data(self, track_csv: str, ego_pose_txt: str):
        """Load tracking data and ego poses"""
        # Load track info
        self.track_df = pd.read_csv(track_csv)
        
        # Filter only vehicles
        self.track_df = self.track_df[self.track_df['object_class'] == 'vehicle'].copy()
        
        # Load ego poses
        with open(ego_pose_txt, 'r') as f:
            self.ego_poses = ast.literal_eval(f.read())
        
        print(f"Loaded {len(self.track_df)} vehicle records")
        print(f"Loaded {len(self.ego_poses)} ego poses")
        print(f"Columns in data: {list(self.track_df.columns)}")
    
    def compute_ego_trajectory(self, start_frame: int, end_frame: int) -> List[Dict]:
        """Compute ego vehicle trajectory from pose matrices with start frame as reference [0, 0]"""
        ego_traj = []
        
        # Get reference pose from start frame
        self.reference_pose = np.array(self.ego_poses[str(start_frame)])
        reference_position = self.reference_pose[:2, 3]  # Reference position
        reference_rotation = self.reference_pose[:2, :2]  # Reference rotation matrix
        
        for frame_id in range(start_frame, end_frame + 1):
            pose = np.array(self.ego_poses[str(frame_id)])
            world_position = pose[:2, 3]  # World position
            
            # Transform to ego-relative coordinates (start frame as origin)
            relative_position = world_position - reference_position
            # Rotate to align with reference frame orientation
            ego_position = np.linalg.inv(reference_rotation) @ relative_position
            # ego_position = relative_position
            
            # Compute speed from consecutive frames (fps = 1 / interval)
            if frame_id > start_frame:
                prev_pose = np.array(self.ego_poses[str(frame_id - 1)])
                prev_world_position = prev_pose[:2, 3]
                distance = np.linalg.norm(world_position - prev_world_position)
                speed = distance / (1 / self.fps)  # fps = frames per second
            else:
                # For first frame, use speed from next frame
                if frame_id < len(self.ego_poses) - 1:
                    next_pose = np.array(self.ego_poses[str(frame_id + 1)])
                    next_world_position = next_pose[:2, 3]
                    distance = np.linalg.norm(next_world_position - world_position)
                    speed = distance / (1 / self.fps)
                else:
                    speed = 0.0
            
            # Extract heading relative to reference frame
            world_rotation = pose[:2, :2]
            relative_rotation = np.linalg.inv(reference_rotation) @ world_rotation
            heading = np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0])
            
            ego_traj.append({
                'frame': frame_id,
                'x': round(ego_position[0], 2),
                'y': round(ego_position[1], 2),
                'heading': round(heading, 2),
                'speed': round(speed, 2)
            })
        
        return ego_traj
    
    def transform_point_current_ego_to_reference_ego(self, frame_id: int, x_in_current_ego: float, y_in_current_ego: float) -> Tuple[float, float]:
        """Transform a point from current-frame ego coordinates to the reference ego frame"""
        current_pose = np.array(self.ego_poses[str(frame_id)])
        current_rotation = current_pose[:2, :2]
        current_position = current_pose[:2, 3]

        reference_rotation = self.reference_pose[:2, :2]
        reference_position = self.reference_pose[:2, 3]
        reference_rotation_inv = np.linalg.inv(reference_rotation)

        point_in_current_ego = np.array([x_in_current_ego, y_in_current_ego], dtype=float)
        point_in_world = current_rotation @ point_in_current_ego + current_position
        point_in_reference_ego = reference_rotation_inv @ (point_in_world - reference_position)

        return float(point_in_reference_ego[0]), float(point_in_reference_ego[1])
    
    def define_collision_zones(self) -> Dict[str, Dict]:
        """
        Define four collision zones around ego vehicle (at origin in ego frame)
        
        Returns:
            Dict with zone names and their boundaries
        """
        zones = {
            'front': {
                'x_min': 0,
                'x_max': self.ego_length,
                'y_min': -self.ego_width / 2,
                'y_max': self.ego_width / 2,
                'description': 'Directly in front of ego vehicle'
            },
            'back': {
                'x_min': -self.ego_length,
                'x_max': 0,
                'y_min': -self.ego_width / 2,
                'y_max': self.ego_width / 2,
                'description': 'Directly behind ego vehicle'
            },
            'left': {
                'x_min': -self.ego_length / 2,
                'x_max': self.ego_length / 2,
                'y_min': self.ego_width + self.ego_width / 2,
                'y_max': self.ego_width / 2,
                'description': 'Left side of ego vehicle'
            },
            'right': {
                'x_min': -self.ego_length / 2,
                'x_max': self.ego_length / 2,
                'y_min': - self.ego_width - self.ego_width / 2,
                'y_max': - self.ego_width / 2,
                'description': 'Right side of ego vehicle'
            }
        }
        
        return zones
    
    def sample_point_in_zone(self, zone_name: str) -> Tuple[float, float]:
        """Randomly sample a collision point within the specified zone"""
        zones = self.define_collision_zones()
        zone = zones[zone_name]
        x = np.random.uniform(zone['x_min'], zone['x_max'])
        y = np.random.uniform(zone['y_min'], zone['y_max'])
        
        return x, y

    # ==============================
    # NEW: user-chosen behavior hook
    # ==============================
    def resolve_behavior_choice(self, behavior_choice: Optional[int or str]) -> str:
        """
        behavior_choice:
          - int: 1-based index into behavior_library (1..N)
          - str: must match (or be a substring of) a library entry
          - None: default to None
        """
        if behavior_choice is None:
            return None

        if isinstance(behavior_choice, int):
            idx = behavior_choice - 1
            if idx < 0 or idx >= len(self.behavior_library):
                raise ValueError(f"behavior_choice int out of range. Expect 1..{len(self.behavior_library)}")
            return self.behavior_library[idx]

        if isinstance(behavior_choice, str):
            s = behavior_choice.strip().lower()
            # exact / substring match
            for b in self.behavior_library:
                if s == b.lower() or s in b.lower():
                    return b
            raise ValueError("behavior_choice string not found in behavior_library (exact or substring match).")

        raise TypeError("behavior_choice must be int, str, or None.")
    
    def prepare_behavior_analyzer_prompt(
        self,
        current_frame: int,
        fps: int,
        history_frames: int,
        selected_behavior: str
    ) -> str:

        # Get historical trajectory data
        start_frame = max(0, current_frame - history_frames)

        # Compute ego vehicle trajectory
        self.ego_traj = self.compute_ego_trajectory(start_frame, current_frame)

        # Collect vehicle trajectories
        self.vehicles_traj = {}
        reference_rotation = self.reference_pose[:2, :2]
        reference_rotation_inv = np.linalg.inv(reference_rotation)

        for frame_id in range(start_frame, current_frame + 1):
            frame_data = self.track_df[self.track_df['frame_id'] == frame_id]

            for _, row in frame_data.iterrows():
                track_id = row['track_id']
                if track_id not in self.vehicles_traj:
                    self.vehicles_traj[track_id] = []

                ego_centered_x = row['box_center_x']
                ego_centered_y = row['box_center_y']

                # Convert vehicle position to the same reference ego frame as ego trajectory
                ref_x, ref_y = self.transform_point_current_ego_to_reference_ego(
                    frame_id=frame_id,
                    x_in_current_ego=float(ego_centered_x),
                    y_in_current_ego=float(ego_centered_y)
                )

                # Estimate velocity from frame-to-frame position changes (in reference ego frame)
                dt = 1.0 / self.fps
                history = self.vehicles_traj[track_id]

                if len(history) > 0:
                    prev_point = history[-1]
                    vx = (ref_x - float(prev_point['x'])) / dt
                    vy = (ref_y - float(prev_point['y'])) / dt
                else:
                    # For the first point, use a forward difference when next frame exists
                    next_frame_data = self.track_df[
                        (self.track_df['frame_id'] == frame_id + 1) &
                        (self.track_df['track_id'] == track_id)
                    ]
                    if not next_frame_data.empty:
                        next_row = next_frame_data.iloc[0]
                        next_ref_x, next_ref_y = self.transform_point_current_ego_to_reference_ego(
                            frame_id=frame_id + 1,
                            x_in_current_ego=float(next_row['box_center_x']),
                            y_in_current_ego=float(next_row['box_center_y'])
                        )
                        vx = (next_ref_x - ref_x) / dt
                        vy = (next_ref_y - ref_y) / dt
                    else:
                        vx = 0.0
                        vy = 0.0

                speed = float(np.hypot(vx, vy))
                heading_ref = float(np.arctan2(vy, vx)) if speed > 1e-6 else 0.0

                self.vehicles_traj[track_id].append({
                    'frame': frame_id,
                    'x': round(ref_x, 2),
                    'y': round(ref_y, 2),
                    'x_in_ego': round(float(ego_centered_x), 2),
                    'y_in_ego': round(float(ego_centered_y), 2),
                    'vx': round(float(vx), 2),
                    'vy': round(float(vy), 2),
                    'heading': round(heading_ref, 2),
                    'speed': round(float(speed), 2)
                })

        zones = self.define_collision_zones()
        zone_descriptions = "\n".join([
            f"- **{name}**: {info['description']} "
            f"(x: [{info['x_min']:.1f}, {info['x_max']:.1f}], "
            f"y: [{info['y_min']:.1f}, {info['y_max']:.1f}])"
            for name, info in zones.items()
        ])

        prompt = f"""# Role
You are an expert safety behavior analyzer for autonomous driving with the ability to CREATE safety-critical scenarios.

# Task Description
Analyze the traffic scenario and identify which background vehicle can be manipulated to create the MOST DANGEROUS collision scenario with the ego vehicle. 


Your job is to:
1) Identify which dangerous behavior from the provided library is most plausible to occur in the current scene.
2) Identify which background vehicle is the MOST plausible / feasible candidate to execute this behavior in the current scene.
3) Given this behavior, select the **target collision zone** (front/back/left/right) that maximizes collision risk with ego.
4) Estimate time-to-collision and risk level.

# Structure of Input Variables
- **Current Frame**: {current_frame}
- **Coordinate System**: World coordinates
- **Background Vehicles**: {len(self.vehicles_traj)} vehicles
- **History Length**: {history_frames} frames
- **Frame Rate**: {self.fps} fps (1 / {self.fps} second per frame)
- **Ego Dimensions**: Length={self.ego_length}m, Width={self.ego_width}m

## Ego Vehicle Trajectory:
Position history: {self.ego_traj}
Coordinate System: World coordinates
Current state: x={self.ego_traj[-1]['x']:.2f}m, y={self.ego_traj[-1]['y']:.2f}m, heading={self.ego_traj[-1]['heading']:.2f}rad, speed={self.ego_traj[-1]['speed']:.2f}m/s

## Collision Zones Definition:
Four rectangular zones tightly surrounding the ego vehicle:
{zone_descriptions}

## Background Vehicle Trajectories:
"""

        for track_id, traj in self.vehicles_traj.items():
            prompt += f"\n### Vehicle {track_id}:\n"
            prompt += f"Current position: x={traj[-1]['x']:.2f}m, y={traj[-1]['y']:.2f}m\n"
            prompt += f"Velocity: vx={traj[-1]['vx']:.2f}m/s, vy={traj[-1]['vy']:.2f}m/s\n"
            prompt += f"Speed: {traj[-1]['speed']:.2f}m/s, Heading: {traj[-1]['heading']:.2f}rad\n"
            prompt += f"Trajectory: {traj}\n"

        prompt += f"""
# Analysis Requirements

## Behavior Library 
{chr(10).join([f"{i+1}. {b}" for i, b in enumerate(self.behavior_library)])}

## Required Analysis Steps

**Step 1: Candidate Filtering**
- Identify which vehicles have the appropriate relative position / lane relationship to plausibly execute:
  the behavior you selected. If a vehicle with adjacent-lane lateral positions relative to ego < 3m, that they are in the same lane, thus could not be selected for "cut-in" behavior. 
- If the behavior implies a "lead vehicle", only consider vehicles ahead of ego in the same lane corridor.
- If the behavior implies "cut-in", prioritize vehicles in adjacent lateral positions with forward overlap potential.
- If the behavior implies "rear tailgating", prioritize vehicles behind ego with high closing speed.

**Step 2: Feasibility & Kinematic Plausibility**
- Based on history (positions/velocities/headings), decide which vehicle can most naturally transition into the target behavior within 3–5 seconds.
- Prefer vehicles whose recent motion already trends toward the required maneuver.

**Step 3: Collision Zone Selection (conditioned on the fixed behavior)**
- Choose the collision zone (front/back/left/right) that best matches the behavior and maximizes severity.
- Consider time-to-zone, closing speed, collision angle, and ego’s ability to evade.

**Step 4: Risk Assessment**
- Assign risk level: "Low", "Medium", "High", or "Critical"
- Provide time-to-collision estimate. The time-to-collision should be estimated under the assumption that the dangerous behavior is happening, not just from the current relative positions and velocities!!! (float seconds)

## Output Requirements
Provide your analysis in JSON format (STRICT):

```json
{{
  "analysis_summary": "Brief explanation of why this vehicle is the best candidate for the fixed behavior and why the zone is most dangerous",
  "selected_vehicle": <track_id>,
  "target_collision_zone": "<front/back/left/right>",
  "behavior_type": "<Dangerous maneuver description>",
  "risk_level": "<Low/Medium/High/Critical>",
  "time_to_collision": "<Estimated time in seconds, as a float, should consider the relative speeds, only consider the time under one direction of collision, not the time to reach the zone>",
  "reasoning": "Detailed explanation why this vehicle is most likely to execute the fixed behavior and why the chosen zone maximizes danger"
}}
```

Respond ONLY with the JSON object.
"""
        return prompt
    
    def call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in autonomous driving safety analysis with creative scenario generation capabilities."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Error: {e}")
            return None
    
    # ==============================
    # MODIFIED: behavior_analyzer now takes user-selected behavior
    # ==============================
    def behavior_analyzer(
        self,
        current_frame: int,
        fps: int,
        history_frames: int,
        behavior_choice: Optional[int or str] = None
    ) -> Optional[Dict]:
        """Behavior analyzer - user selects behavior, LLM selects vehicle and collision zone"""
        selected_behavior = self.resolve_behavior_choice(behavior_choice)

        print(f"\n{'='*60}")
        print(f"Analyzing Frame {current_frame}")
        print(f"{'='*60}")
        print(f"User-selected behavior: {selected_behavior}")

        prompt = self.prepare_behavior_analyzer_prompt(
            current_frame=current_frame,
            fps=fps,
            history_frames=history_frames,
            selected_behavior=selected_behavior
        )

        print("Querying LLM Behavior Analyzer...")
        response = self.call_llm(prompt, temperature=0.3)

        if not response:
            return None

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response.strip())

            # Ensure behavior fields are consistent with user choice (hard-override for safety)
            # result["selected_behavior"] = selected_behavior
            # result["behavior_type"] = selected_behavior

            print(f"\n✓ Analysis Complete:")
            print(f"  - Selected Vehicle: {result['selected_vehicle']}")
            print(f"  - Target Collision Zone: {result['target_collision_zone']}")
            print(f"  - Behavior: {result['behavior_type']}")
            print(f"  - Risk Level: {result['risk_level']}")
            print(f"  - Time to Collision: {result['time_to_collision']} seconds")
            print(f"  - Reasoning: {result['reasoning']}")

            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response}")
            return None

    def feasible_trajectory_generation(
        self,
        behavior_analysis: Dict,
        current_frame: int,
        history_frames: int,
        fps: int
    ) -> Optional[List[Dict]]:
        """
        New design:
        - LLM ONLY generates spatio-temporal waypoints [(t, x, y)]
        - No kinematics, no velocity, no feasibility constraints
        - Trajectory is a pure reference path for CARLA PID controller
        """

        print(f"\n{'='*60}")
        print("Reference Trajectory Generation (LLM → Waypoints Only)")
        print(f"{'='*60}")

        track_id = behavior_analysis["selected_vehicle"]
        target_zone = behavior_analysis["target_collision_zone"]
        horizon_sec = float(behavior_analysis["time_to_collision"])
        behavior_choice = behavior_analysis["behavior_type"]

        # ------------------------------------------------------------------
        # 1. Current vehicle state
        # ------------------------------------------------------------------
        frame_data = self.track_df[
            (self.track_df["frame_id"] == current_frame) &
            (self.track_df["track_id"] == track_id)
        ]

        if frame_data.empty:
            print("✗ Vehicle not found in current frame")
            return None

        row = frame_data.iloc[0]

        start_x, start_y = self.transform_point_current_ego_to_reference_ego(
            frame_id=current_frame,
            x_in_current_ego=float(row["box_center_x"]),
            y_in_current_ego=float(row["box_center_y"])
        )

        # update ego future
        end_frame = current_frame + int(horizon_sec * fps)
        start_frame = max(0, current_frame - history_frames)
        self.ego_traj_future = self.compute_ego_trajectory(start_frame, end_frame)

        # ------------------------------------------------------------------
        # 2. Get vehicle's historical trajectory for continuity
        # ------------------------------------------------------------------
        vehicle_history = self.vehicles_traj.get(int(track_id), [])
        if len(vehicle_history) < 2:
            print("✗ Insufficient vehicle history for continuous prediction")
            return None
        
        # Sample FINAL collision point (fixed, not optimized)
        target_x, target_y = self.sample_point_in_zone(target_zone)
        # map to absolute coordinates (relative to ego's future position)
        target_x += self.ego_traj_future[-1]['x']
        target_y += self.ego_traj_future[-1]['y']

        # ------------------------------------------------------------------
        # 3. Prompt LLM to generate continuous waypoints
        # ------------------------------------------------------------------
        num_steps = int(horizon_sec * fps)
        dt = 1.0 / fps

        # Prepare vehicle history for context
        history_str = "\n".join([
            f"Frame {h['frame']}: x={h['x']:.2f}, y={h['y']:.2f}, vx={h['vx']:.2f}, vy={h['vy']:.2f}, speed={h['speed']:.2f}"
            for h in vehicle_history
        ])
        
        # Prepare ego trajectory for context
        ego_traj_str = "\n".join([
            f"t={i*dt:.2f}s: x={p['x']:.2f}, y={p['y']:.2f}, speed={p['speed']:.2f}"
            for i, p in enumerate(self.ego_traj_future)
        ])

        prompt = f"""
    You are generating a CONTINUOUS REFERENCE TRAJECTORY for a vehicle to create a traffic accident: '{behavior_choice}'. 

    Vehicle Context:
    ## Recent History:
    {history_str}

    ## Current State:
    Position: x={start_x:.2f}, y={start_y:.2f}

    ## Ego Vehicle Trajectory (AVOID EARLY COLLISION):
    {ego_traj_str}

    ## Target:
    Final collision point: x={target_x:.2f}, y={target_y:.2f}
    (Vehicle MUST end at this point at the end of trajectory)
                                          
    Task:
    Generate a trajectory that:
    0. (Most Important!) Connect the current vehicle position to the final collision point ({target_x:.2f}, {target_y:.2f})
    1. The trajectory MUST align with the selected dangerous behavior: '{behavior_choice}'. For example, if the behavior is "cut-in from adjacent lane", ensure the trajectory reflects a lateral movement into ego's lane
    2. Gradually transitions to aim toward the collision point, no sudden turnings
    3. Starts from current position with CONTINUOUS continuation of current motion
    4. AVOIDS premature collision with ego vehicle trajectory (maintain safe distance until final approach)
    5. Contains exactly {num_steps} waypoints over {horizon_sec:.1f} seconds
    6. Do NOT explain anything, just output JSON array
    7. You ONLY output time-stamped (x, y) waypoints

    Time Parameters:
    - Total duration = {horizon_sec:.1f} seconds
    - Timestep = {dt:.2f} seconds
    - Frame rate = {fps} fps

    Output format (STRICT JSON, no comments):

    [
    {{ "t": 0.0, "x": {start_x:.2f}, "y": {start_y:.2f} }},
    {{ "t": {dt:.2f}, "x": ..., "y": ... }},
    ...
    {{ "t": {horizon_sec - dt:.2f}, "x": ..., "y": ... }}
    ]

    """

        print("Querying LLM for reference waypoints...")
        response = self.call_llm(prompt, temperature=0.2)

        if response is None:
            print("✗ LLM failed")
            return None

        try:
            if "```" in response:
                response = response.split("```")[1]
            waypoints = json.loads(response)
        except Exception as e:
            print("✗ Failed to parse LLM output")
            print(response)
            return None

        # ------------------------------------------------------------------
        # 4. Convert to internal trajectory format
        # ------------------------------------------------------------------
        trajectory = []
        for wp in waypoints:
            trajectory.append({
                "t": float(wp["t"]),
                "x": float(wp["x"]),
                "y": float(wp["y"])
            })

        print(f"✓ Generated {len(trajectory)} reference waypoints")
        print(trajectory)
        print(target_x, target_y)

        # ------------------------------------------------------------------
        # 5. Visualization: color-coded by time (cold → warm)
        # ------------------------------------------------------------------
        # times = np.array([p["t"] for p in trajectory])
        # xs = np.array([p["x"] for p in trajectory])
        # ys = np.array([p["y"] for p in trajectory])

        # cmap = plt.cm.plasma
        # norm = plt.Normalize(times.min(), times.max())
        # colors = cmap(norm(times))

        # plt.figure(figsize=(12, 8))
        
        # # Plot vehicle's historical trajectory for context
        # hist_xs = np.array([p["x"] for p in vehicle_history])
        # hist_ys = np.array([p["y"] for p in vehicle_history])
        # plt.plot(hist_xs, hist_ys, 'c--', linewidth=2, alpha=0.6)
        # plt.scatter(hist_xs, hist_ys, c="cyan", s=20, alpha=0.6)
        
        # # Plot generated vehicle trajectory
        # plt.scatter(xs, ys, c=colors, s=40, edgecolors="k")
        # plt.plot(xs, ys, alpha=0.7, color="blue", linewidth=2)

        # # Plot ego trajectory (past)
        # ego_xs = np.array([p["x"] for p in self.ego_traj])
        # ego_ys = np.array([p["y"] for p in self.ego_traj])
        # plt.plot(ego_xs, ego_ys, 'g-', linewidth=3, alpha=0.7)
        # plt.scatter(ego_xs, ego_ys, c="green", s=30, alpha=0.8)
        
        # # Plot ego future trajectory
        # ego_future_xs = np.array([p["x"] for p in self.ego_traj_future])
        # ego_future_ys = np.array([p["y"] for p in self.ego_traj_future])
        # plt.plot(ego_future_xs, ego_future_ys, 'g:', linewidth=3, alpha=0.5)
        # plt.scatter(ego_future_xs, ego_future_ys, c="lightgreen", s=20, alpha=0.6)

        # # Plot key points
        # plt.scatter(0, 0, c="red", s=120, marker="*")
        # plt.scatter(xs[0], ys[0], c="blue", s=100, marker="o")
        # plt.scatter(xs[-1], ys[-1], c="black", s=100, marker="X")
        # plt.scatter(target_x, target_y, c="Green", s=200, marker="X")

        # # Add collision zones visualization
        # zones = self.define_collision_zones()
        # for zone_name, zone_info in zones.items():
        #     x_min, x_max = zone_info['x_min'], zone_info['x_max']
        #     y_min, y_max = zone_info['y_min'], zone_info['y_max']
        #     plt.plot([x_min, x_max, x_max, x_min, x_min], 
        #             [y_min, y_min, y_max, y_max, y_min], 
        #             '--', alpha=0.5)

        # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
        # cbar.set_label("Time (s)")

        # plt.title(f"Reference Trajectory (Vehicle {track_id}) with Ego Path", fontsize=20)
        # plt.xlabel("X (m)")
        # plt.ylabel("Y (m)")
        # plt.grid(True)
        # plt.axis("equal")
        # plt.tight_layout()
        # plt.show()

        return trajectory

    def save_corner_case_scenario(
        self,
        analysis_frame: int,
        behavior_result: Dict,
        adversarial_trajectory: List[Dict],
        output_file: str
    ):
        """Save corner case scenario data to CSV file including ego trajectory"""
        
        # Get selected vehicle's history
        track_id = behavior_result["selected_vehicle"]
        vehicle_history = self.vehicles_traj.get(int(track_id), [])
        
        # Prepare data for CSV
        csv_data = []
        
        # Add vehicle history data with timesteps (convert frame to time)
        # for i, hist_point in enumerate(vehicle_history):
        #     # Calculate timestep
        #     # Make timesteps relative to start of trajectory
        #     timestep = i * (1 / self.fps)  # 1 / fps second per frame
            
        #     csv_data.append({
        #         'timestep': timestep,
        #         'x': hist_point['x'],
        #         'y': hist_point['y'],
        #         'heading': 0.0, # don't care
        #         'trajectory_type': 'target_history',
        #         'vehicle_id': track_id,
        #         'frame_id': hist_point['frame']
        #     })
        
        # Add adversarial trajectory data (skip the first frame to avoid duplication)
        # for traj_point in adversarial_trajectory[1:]:
        for traj_point in adversarial_trajectory[0:]:
            # Offset adversarial trajectory timesteps to continue after history
            # History ends at (len(vehicle_history) - 1) * (1 / self.fps), so start from there
            # offset_timestep = (len(vehicle_history) - 1) * (1 / self.fps) + traj_point['t']
            offset_timestep = traj_point['t']
            
            csv_data.append({
                'timestep': offset_timestep,
                'x': traj_point['x'],
                'y': traj_point['y'], 
                'heading': 0.0, # don't care
                'trajectory_type': 'target_future',
                'vehicle_id': track_id,
                'frame_id': analysis_frame + int(traj_point['t'] * self.fps),  # Convert time back to frame
                'behavior_type': behavior_result["behavior_type"],
                'zone': behavior_result["target_collision_zone"]
            })
        
        # Add ego vehicle historical trajectory (aligned with vehicle history timeline)
        # for i, ego_point in enumerate(self.ego_traj):
        #     timestep = i * (1 / self.fps)  # Same timeline as vehicle history
            
        #     csv_data.append({
        #         'timestep': timestep,
        #         'x': ego_point['x'],
        #         'y': ego_point['y'],
        #         'heading': ego_point['heading'],
        #         'trajectory_type': 'ego_history',
        #         'vehicle_id': 'ego',
        #         'frame_id': ego_point['frame']
        #     })
        
        # Add ego vehicle future trajectory (aligned with adversarial trajectory timeline)
        # Start from where ego history ends to avoid duplication
        # ego_history_end_time = (len(self.ego_traj) - 1) * (1 / self.fps)
        
        # for i, ego_point in enumerate(self.ego_traj_future[len(self.ego_traj):]):
        for i, ego_point in enumerate(self.ego_traj_future[len(self.ego_traj)-1:]):
            # Calculate timestep continuing from ego history end
            # timestep = ego_history_end_time + (i + 1) * (1 / self.fps)
            timestep = i * (1 / self.fps)
            
            csv_data.append({
                'timestep': timestep,
                'x': ego_point['x'],
                'y': ego_point['y'],
                'heading': ego_point['heading'],
                'trajectory_type': 'ego_future',
                'vehicle_id': 'ego',
                'frame_id': ego_point['frame'],
                'behavior_type': behavior_result["behavior_type"],
                'zone': behavior_result["target_collision_zone"]
            })
        
        # Convert to DataFrame and sort by timestep
        import pandas as pd
        df = pd.DataFrame(csv_data)
        df = df.sort_values(['vehicle_id', 'timestep']).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved corner case scenario to {output_file}")
        print(f"  - {len(vehicle_history)} original vehicle trajectory points")
        print(f"  - {len(adversarial_trajectory)} adversarial vehicle trajectory points") 
        print(f"  - {len(self.ego_traj)} ego history trajectory points")
        print(f"  - {len(self.ego_traj_future) - len(self.ego_traj)} ego future trajectory points")
        print(f"  - Total duration: {df['timestep'].max():.1f} seconds")
        print(f"  - Total records: {len(df)} (ego + target vehicle)")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
    def run_full_pipeline(
        self,
        analysis_frame: int,
        output_file: str = "corner_track_info.csv",
        history_frames: int = 30,
        fps: int = 10,
        behavior_choice: Optional[int or str] = None
    ):
        """Run the complete improved pipeline (user selects behavior)"""

        print("\n" + "="*80)
        print(" IMPROVED SAFETY-CRITICAL SCENARIO GENERATION PIPELINE")
        print(" Innovation: Collision Zone Targeting Approach")
        print("="*80)

        # Step 1: Behavior Analysis (user-chosen behavior -> LLM picks vehicle + zone)
        behavior_result = self.behavior_analyzer(
            analysis_frame,
            fps=fps,
            history_frames=history_frames,
            behavior_choice=behavior_choice
        )

        if not behavior_result:
            print("\n✗ Pipeline failed: Behavior analysis unsuccessful")
            return None

        # Step 2: Collision-Oriented Trajectory Generation
        adversarial_trajectory = self.feasible_trajectory_generation(
            behavior_result,
            analysis_frame,
            history_frames,
            fps=fps
        )

        if not adversarial_trajectory:
            print("\n✗ Pipeline failed: Trajectory generation unsuccessful")
            return None

        # Step 3: Save Results
        self.save_corner_case_scenario(
            analysis_frame,
            behavior_result,
            adversarial_trajectory,
            output_file
        )

        print("\n" + "="*80)
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)

        return {
            'behavior_analysis': behavior_result,
            'trajectory': adversarial_trajectory,
            'output_file': output_file
        }



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-agent LLM corner-case trajectory generation.")
    parser.add_argument("--scene", default="134", help="scene id; reads <data_dir>/<scene>/track_info.csv and ego_pose.txt")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--behavior", default=None, help="behavior from the library (e.g. sudden braking); None lets the LLM choose")
    args = parser.parse_args()

    fps = args.fps
    scene = args.scene
    scene_dir = os.path.join(args.data_dir, scene)
    behavior_choice = args.behavior

    # Initialize improved generator
    API_KEY = os.environ["OPENAI_API_KEY"]
    generator = ImprovedSafetyCriticalGenerator(
        fps=fps,
        api_key=API_KEY,
        model=args.model
    )
    
    # Customize ego vehicle dimensions if needed
    generator.ego_length = 4.5  # meters
    generator.ego_width = 2.0   # meters
    # horizon_sec = 3.0
    
    # Load data
    generator.load_data(
        track_csv=os.path.join(scene_dir, "track_info.csv"),
        ego_pose_txt=os.path.join(scene_dir, "ego_pose.txt")
    )

    # read the length of track
    max_frame_id = generator.track_df['frame_id'].max()

    # iterate through frames to generate corner case scenarios at different time points
    for frame_id in range(90, max_frame_id - 60, 10):
        print(f"\nProcessing frame {frame_id}/{max_frame_id}...")
        history_sec = float(frame_id / fps)
        history_frames = int(history_sec * fps)

        result = generator.run_full_pipeline(
            analysis_frame=frame_id,
            output_file=os.path.join(scene_dir, f"corner_track_info__{scene}_{frame_id}.csv"),
            history_frames=history_frames,
            fps=fps,
            behavior_choice=behavior_choice
        )
    