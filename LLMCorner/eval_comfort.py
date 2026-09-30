import numpy as np
import pandas as pd
import glob


# ============================================================
# Fixed timestep (true sampling rate)
# ============================================================

DT = 0.1


# ============================================================
# Keep middle frames
# ============================================================

def keep_middle_70(df):

    n = len(df)

    # if n < 10 :
    #     raise ValueError("trajectory too short")
    # if n > 30 :
    #     raise ValueError("trajectory too long")

    start = int(np.floor(n * 0.30))
    end   = int(np.ceil(n * 0.60))

    return df.iloc[start:end].reset_index(drop=True)


# ============================================================
# Finite difference with fixed dt
# ============================================================

def finite_diff(y, dt=DT):

    y = np.asarray(y)

    dy = np.diff(y)

    v = dy / dt

    if len(v) == 0:
        return np.zeros_like(y)

    return np.concatenate([v, [v[-1]]])


# ============================================================
# Velocity estimation
# ============================================================

def velocity_from_position(x, y):

    vx = finite_diff(x)
    vy = finite_diff(y)

    return vx, vy


def velocity_from_speed_heading(speed, heading):

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    return vx, vy


# ============================================================
# Higher order derivatives
# ============================================================

def acceleration(vx, vy):

    ax = finite_diff(vx)
    ay = finite_diff(vy)

    return ax, ay


def jerk(ax, ay):

    jx = finite_diff(ax)
    jy = finite_diff(ay)

    return jx, jy


# ============================================================
# Curvature
# ============================================================

def curvature(vx, vy, ax, ay):

    num = vx * ay - vy * ax
    den = (vx**2 + vy**2)**1.5 + 1e-6

    return num / den


# ============================================================
# Compute metrics
# ============================================================

def compute_metrics(csv_path):

    df = pd.read_csv(csv_path)

    tgt = df[df["trajectory_type"] == "target_future"].copy()
    tgt = tgt.sort_values("timestep")
    tgt = keep_middle_70(tgt)

    heading = tgt["heading"].values

    # ============================================================
    # Velocity source selection
    # ============================================================

    if "speed" in tgt.columns and not tgt["speed"].isna().all():

        # CARLA trajectory
        speed = tgt["speed"].values
        vx, vy = velocity_from_speed_heading(speed, heading)

    else:

        # LLM-only trajectory
        x = tgt["x"].values
        y = tgt["y"].values

        vx, vy = velocity_from_position(x, y)

    # ============================================================
    # Acceleration
    # ============================================================

    ax, ay = acceleration(vx, vy)

    # ============================================================
    # Jerk
    # ============================================================

    jx, jy = jerk(ax, ay)
    j = np.sqrt(jx**2 + jy**2)

    # ============================================================
    # Lateral acceleration
    # ============================================================

    speed_safe = np.sqrt(vx**2 + vy**2) + 1e-6

    a_lat = (vx * ay - vy * ax) / speed_safe

    # ============================================================
    # Curvature
    # ============================================================

    kappa = curvature(vx, vy, ax, ay)

    mask = speed_safe > 1.0
    kappa[~mask] = 0

    kappa_dot = finite_diff(kappa)

    # ============================================================
    # Percentiles
    # ============================================================

    a_lat_95 = np.percentile(np.abs(a_lat), 95)
    j_95 = np.percentile(np.abs(j), 95)
    kappa_dot_95 = np.percentile(np.abs(kappa_dot), 95)

    # ============================================================
    # Comfort violation
    # ============================================================

    viol = (
        (np.abs(a_lat) > 3.0)
    )

    comfort_viol = np.mean(viol) * 100

    return {

        "a_lat_95": a_lat_95,
        "j_95": j_95,
        "kappa_dot_95": kappa_dot_95,
        "comfort_viol": comfort_viol

    }


# ============================================================
# Batch evaluation
# ============================================================

def evaluate_folder(csv_files):

    rows = []

    for f in csv_files:

        try:
            rows.append(compute_metrics(f))

        except:
            print("[skip]", f)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

llm_files = glob.glob("data/all_eval_review/*.csv")
carla_files = glob.glob("data/all_eval_CARLA_review/*.csv")

df_llm = evaluate_folder(llm_files)
df_carla = evaluate_folder(carla_files)


print("\nLLM-only")
print(df_llm.describe().loc[["mean","std"]])

print("\nLLM+CARLA")
print(df_carla.describe().loc[["mean","std"]])