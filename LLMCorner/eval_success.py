import numpy as np
import pandas as pd
import glob

# ============================================================
# 1) Zone definition
# ============================================================

def define_collision_zones(ego_length, ego_width):

    zones = {

        "front": {
            "x_min": 0,
            "x_max": ego_length,
            "y_min": -ego_width/2,
            "y_max": ego_width/2
        },

        "back": {
            "x_min": -ego_length,
            "x_max": 0,
            "y_min": -ego_width/2,
            "y_max": ego_width/2
        },

        "left": {
            "x_min": -ego_length/2,
            "x_max": ego_length/2,
            "y_min": ego_width/2,
            "y_max": ego_width + ego_width/2
        },

        "right": {
            "x_min": -ego_length/2,
            "x_max": ego_length/2,
            "y_min": -ego_width - ego_width/2,
            "y_max": -ego_width/2
        }

    }

    return zones


def point_in_zone(x, y, zone):

    return (
        zone["x_min"] <= x <= zone["x_max"]
        and zone["y_min"] <= y <= zone["y_max"]
    )


# ============================================================
# 2) TTC computation
# ============================================================

def finite_diff(y, t):

    dy = np.diff(y)
    dt = np.diff(t)

    v = dy / np.maximum(dt, 1e-6)

    return np.concatenate([v, [v[-1]]])


def compute_velocity(x, y, t):

    vx = finite_diff(x, t)
    vy = finite_diff(y, t)

    return np.stack([vx, vy], axis=1)


def compute_ttc(p_ego, p_adv, v_ego, v_adv):

    r = p_adv - p_ego
    dist = np.linalg.norm(r, axis=1)

    v_rel = v_adv - v_ego

    closing_speed = -np.sum(r * v_rel, axis=1) / (dist + 1e-6)

    ttc = np.full_like(dist, np.inf)

    mask = closing_speed > 0

    ttc[mask] = dist[mask] / closing_speed[mask]

    ttc = np.minimum(ttc, 5.0)

    return ttc, dist


# ============================================================
# 3) Rule-based baseline
# ============================================================

def generate_rule_based(
    ego_df,
    tgt_df,
    dt=0.1
):

    ego = ego_df.iloc[0]
    tgt = tgt_df.iloc[0]

    rel_x = tgt["x"] - ego["x"]
    rel_y = tgt["y"] - ego["y"]

    # choose behavior
    if rel_x > 0 and abs(rel_y) < 2:
        behavior = "brake"

    elif rel_x < 0 and abs(rel_y) < 2:
        behavior = "tailgate"

    else:
        behavior = "cutin"

    t = tgt_df["timestep"].values
    n = len(t)

    x = np.zeros(n)
    y = np.zeros(n)

    x[0] = tgt["x"]
    y[0] = tgt["y"]

    dt = tgt_df["timestep"].iloc[1] - tgt_df["timestep"].iloc[0]
    dx = tgt_df["x"].iloc[1] - tgt_df["x"].iloc[0]
    dy = tgt_df["y"].iloc[1] - tgt_df["y"].iloc[0]
    v = np.sqrt(dx**2 + dy**2) / dt

    for i in range(1, n):

        if behavior == "brake":

            v = max(v - 3*dt, 0)

            x[i] = x[i-1] + v*dt
            y[i] = y[i-1]

        elif behavior == "tailgate":

            v = v + 2.0*dt

            x[i] = x[i-1] + v*dt
            y[i] = y[i-1]

        elif behavior == "cutin":

            x[i] = x[i-1] + v*dt

            y_shift = -rel_y * 1.0

            y[i] = y[i-1] + y_shift

    traj = pd.DataFrame({

        "timestep": t,
        "x": x,
        "y": y

    })

    return traj

# ============================================================
# 4) Random baseline
# ============================================================

def generate_random(tgt_df):

    t = tgt_df["timestep"].values
    n = len(t)

    start_x = tgt_df.iloc[0]["x"]
    start_y = tgt_df.iloc[0]["y"]

    xs = [start_x]
    ys = [start_y]

    for i in range(1, n):

        dx = np.random.normal(0, 0.8)
        dy = np.random.normal(0, 0.8)

        xs.append(xs[-1] + dx)
        ys.append(ys[-1] + dy)

    traj = pd.DataFrame({
        "timestep": t,
        "x": xs,
        "y": ys
    })

    return traj


# ============================================================
# 5) Metrics
# ============================================================

def evaluate_single(csv_path, ego_length, ego_width, method):

    df = pd.read_csv(csv_path)

    ego = df[df["trajectory_type"] == "ego_future"].copy()
    tgt = df[df["trajectory_type"] == "target_future"].copy()

    behavior = df["behavior_type"].iloc[0]
    z_star = df["zone"].iloc[0]

    # # Filter out short trajectories
    if len(ego) < 5 or len(tgt) < 5:
        raise ValueError(f"Trajectory too short (ego={len(ego)}, tgt={len(tgt)})")

    # # Keep middle 70% of each trajectory
    # def _middle_70(df_in):
    #     n = len(df_in)
    #     lo = int(np.floor(n * 0.15))
    #     hi = int(np.ceil(n * 0.85))
    #     return df_in.iloc[lo:hi].reset_index(drop=True)

    # ego = _middle_70(ego.sort_values("timestep"))
    # tgt = _middle_70(tgt.sort_values("timestep"))

    if method == "LLM":
        adv = tgt
    elif method == "Rule":
        adv = generate_rule_based(ego, tgt)
    elif method == "Random":
        adv = generate_random(tgt)

    # Build a common time grid from the intersection of ego and adv timesteps
    t_ego = ego["timestep"].values.astype(float)
    t_adv = adv["timestep"].values.astype(float)
    t_min = max(t_ego[0], t_adv[0])
    t_max = min(t_ego[-1], t_adv[-1])
    if t_max <= t_min + 1e-6:
        raise ValueError("Ego and adv time ranges do not overlap.")

    dt = min(
        float(np.median(np.diff(t_ego))) if len(t_ego) > 1 else 0.04,
        float(np.median(np.diff(t_adv))) if len(t_adv) > 1 else 0.04,
    )
    t = np.arange(t_min, t_max + 1e-9, dt)

    ego_x = np.interp(t, t_ego, ego["x"].values.astype(float))
    ego_y = np.interp(t, t_ego, ego["y"].values.astype(float))
    adv_x = np.interp(t, t_adv, adv["x"].values.astype(float))
    adv_y = np.interp(t, t_adv, adv["y"].values.astype(float))

    p_ego = np.stack([ego_x, ego_y], axis=1)
    p_adv = np.stack([adv_x, adv_y], axis=1)

    v_ego = compute_velocity(ego_x, ego_y, t)
    v_adv = compute_velocity(adv_x, adv_y, t)

    ttc, dist = compute_ttc(p_ego, p_adv, v_ego, v_adv)

    min_ttc = float(np.min(ttc))

    idx = int(np.argmin(dist))

    zones = define_collision_zones(ego_length, ego_width)

    dx = p_adv[idx, 0] - p_ego[idx, 0]
    dy = p_adv[idx, 1] - p_ego[idx, 1]

    hit = point_in_zone(dx, dy, zones[z_star])

    success = min_ttc < 1.0

    time_to_event = float(t[int(np.argmin(ttc))] - t[0])

    return {
        "behavior_type": behavior,
        "z_star": z_star,
        "ZoneHit": int(hit),
        "Success": int(success),
        "BehaviorComp": int(method == "LLM"),
        "MinTTC(s)": min_ttc,
        "Time-to-Event(s)": time_to_event,
        "Method": method,
    }


# ============================================================
# 6) Batch evaluation
# ============================================================

def evaluate_all(csv_files, ego_length, ego_width):

    rows = []

    for f in csv_files:

        for m in ["LLM","Rule","Random"]:

            try:
                # print(f"Evaluating {f} with method {m}...")
                rows.append(evaluate_single(f, ego_length, ego_width, m))

            except:

                pass

    return pd.DataFrame(rows)


# ============================================================
# 7) Run evaluation
# ============================================================

csv_files = glob.glob("data/all_eval/*.csv")

df = evaluate_all(csv_files, 4.5, 2.0)

print(df)

print("\n=== Mean metrics by method ===")

stats = df.groupby("Method")[[
    "ZoneHit",
    "Success",
    "MinTTC(s)",
]].agg(["mean","std"])

print(stats)