"""
Module 1 — movement feature extraction (speed, acceleration, jump height,
court zones, MEI, etc). Content moved verbatim out of the original
api-ready notebook's cells 2 (CONF_THR), 5 (small classification helpers),
and 8 (extract_movement_features).
"""
import json
import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

# Confidence threshold used when deciding whether a keypoint is reliable
# enough to feed into movement calculations (notebook cell 2).
CONF_THR = 0.25


# ------------------------------------------------------------------
# Small classification helpers (notebook cell 5)
# ------------------------------------------------------------------
def _movement_direction(dx_px, dy_px):
    """Angle in degrees. 0 = right, CCW positive."""
    if abs(dx_px) < 1e-3 and abs(dy_px) < 1e-3:
        return 0.0
    angle = math.degrees(math.atan2(-dy_px, dx_px))
    return round(angle % 360, 1)


def _classify_step(jump_px, speed, accel):
    if jump_px > 15:
        return 'jump'
    if speed > 4.0:
        return 'sprint'
    if abs(accel) > 5.0:
        return 'lunge'
    if speed > 1.0:
        return 'step'
    return 'stand'


# ------------------------------------------------------------------
# Main entry point (notebook cell 8)
# ------------------------------------------------------------------
def extract_movement_features(input_json_path, output_json_path, output_csv_path,
                               output_video_path=None, original_video_path=None,
                               match_id="match_001", player_id="player_01"):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    FPS        = data.get('fps', 30)
    FRAME_W    = data.get('frame_width', 1920)
    FRAME_H    = data.get('frame_height', 1080)
    frames_raw = data.get('frames', [])

    # Pixels → metres calibration (standard badminton court)
    COURT_M_HEIGHT = 13.4
    COURT_M_WIDTH  = 6.1
    PX_PER_M_Y = FRAME_H / COURT_M_HEIGHT
    PX_PER_M_X = FRAME_W / COURT_M_WIDTH

    # ── Build per-frame DataFrame (detected frames only) ──────────────────────
    rows = []
    for fr in frames_raw:
        if not fr.get('player_detected', False):
            continue

        bb  = fr['bounding_box']
        kps = {kp['name']: kp for kp in fr.get('keypoints', [])}

        def kp_xy(name):
            k = kps.get(name)
            if k and k['confidence'] >= CONF_THR:
                return k['x'], k['y']
            return np.nan, np.nan

        lax, lay = kp_xy('left_ankle')
        rax, ray = kp_xy('right_ankle')
        foot_x   = np.nanmean([lax, rax])
        foot_y   = np.nanmean([lay, ray])

        lhx, lhy = kp_xy('left_hip')
        rhx, rhy = kp_xy('right_hip')
        hip_x    = np.nanmean([lhx, rhx])
        hip_y    = np.nanmean([lhy, rhy])

        lwx, lwy = kp_xy('left_wrist')
        rwx, rwy = kp_xy('right_wrist')

        rows.append(dict(
            frame_id=fr['frame_id'],
            timestamp=fr['timestamp'],
            bb_x1=bb['x1'], bb_y1=bb['y1'], bb_x2=bb['x2'], bb_y2=bb['y2'],
            bb_w=bb['width'], bb_h=bb['height'],
            foot_x=foot_x, foot_y=foot_y,
            hip_x=hip_x,   hip_y=hip_y,
            lax=lax, lay=lay, rax=rax, ray=ray,
            lwx=lwx, lwy=lwy, rwx=rwx, rwy=rwy,
            lhx=lhx, lhy=lhy, rhx=rhx, rhy=rhy,
        ))

    df = pd.DataFrame(rows).sort_values('frame_id').reset_index(drop=True)
    if len(df) == 0:
        return {}

    # ── Smoothing window (≥3, odd, ~100 ms) ───────────────────────────────────
    WIN = max(3, int(FPS * 0.10))
    WIN = WIN if WIN % 2 == 1 else WIN + 1

    df['foot_x_sm'] = savgol_filter(df['foot_x'].ffill().bfill(), WIN, 2)
    df['foot_y_sm'] = savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)

    # ── Speed ──────────────────────────────────────────────────────────────────
    dx_px = df['foot_x_sm'].diff()
    dy_px = df['foot_y_sm'].diff()
    dt    = df['timestamp'].diff().replace(0, np.nan)

    dx_m = dx_px / PX_PER_M_X
    dy_m = dy_px / PX_PER_M_Y

    df['speed_mps'] = np.sqrt(dx_m**2 + dy_m**2) / dt
    df['speed_mps'] = df['speed_mps'].clip(upper=15)

    # ── Acceleration (store in df to share the same index as dt) ──────────────
    df['speed_sm']  = savgol_filter(df['speed_mps'].fillna(0), WIN, 2)
    df['accel_mps2'] = df['speed_sm'].diff() / dt
    df['accel_mps2'] = df['accel_mps2'].clip(-30, 30)

    # ── Court zone ────────────────────────────────────────────────────────────
    def lateral_zone(x):
        if pd.isna(x): return 'Unknown'
        frac = x / FRAME_W
        if frac < 0.33:   return 'Left'
        elif frac < 0.67: return 'Centre'
        else:             return 'Right'

    def depth_zone(y):
        if pd.isna(y): return 'Unknown'
        frac = y / FRAME_H
        if frac < 0.33:   return 'Net'
        elif frac < 0.67: return 'Mid'
        else:             return 'Back'

    df['lateral_zone'] = df['foot_x'].apply(lateral_zone)
    df['depth_zone']   = df['foot_y'].apply(depth_zone)
    df['court_zone']   = df['lateral_zone'] + '-' + df['depth_zone']

    # ── Stride frequency ───────────────────────────────────────────────────────
    left_y  = df['lay'].ffill().bfill().values
    right_y = df['ray'].ffill().bfill().values
    min_dist_frames = int(FPS * 0.15)

    left_peaks,  _ = find_peaks(-left_y,  distance=min_dist_frames, prominence=5)
    right_peaks, _ = find_peaks(-right_y, distance=min_dist_frames, prominence=5)
    total_peaks    = len(left_peaks) + len(right_peaks)
    total_seconds  = df['timestamp'].max() - df['timestamp'].min()

    all_peaks  = sorted(np.concatenate([left_peaks, right_peaks]))
    peak_times = df['timestamp'].iloc[all_peaks].values

    df['stride_freq_hz'] = np.nan
    for i, row in df.iterrows():
        t = row['timestamp']
        count = np.sum((peak_times >= t - 0.5) & (peak_times <= t + 0.5))
        df.at[i, 'stride_freq_hz'] = count

    # ── Jump height ───────────────────────────────────────────────────────────
    hip_y_series   = df['hip_y'].ffill().bfill()
    baseline_window = max(3, int(FPS))
    hip_baseline   = hip_y_series.rolling(baseline_window, center=True, min_periods=1).median()
    df['jump_height_px'] = (hip_baseline - hip_y_series).clip(lower=0)

    jump_peaks, _ = find_peaks(
        df['jump_height_px'],
        height=10,
        distance=int(FPS * 0.3),
        prominence=8
    )

    # ── Burst / recovery speed ────────────────────────────────────────────────
    BURST_THRESHOLD = df['speed_mps'].quantile(0.75)
    POST_FRAMES     = int(0.5 * FPS)
    burst_frames, _ = find_peaks(
        df['speed_mps'].fillna(0),
        height=BURST_THRESHOLD,
        distance=int(FPS * 0.3)
    )

    df['is_burst'] = False
    df.loc[df.index[burst_frames], 'is_burst'] = True

    # ── Lateral reach ─────────────────────────────────────────────────────────
    df['lat_reach_px']   = abs(df['rwx'] - df['lwx'])
    df['lat_reach_m']    = df['lat_reach_px'] / PX_PER_M_X
    df['hip_cx']         = (df['lhx'].fillna(df['rhx']) + df['rhx'].fillna(df['lhx'])) / 2
    df['left_reach_m']   = abs(df['lwx'] - df['hip_cx']) / PX_PER_M_X
    df['right_reach_m']  = abs(df['rwx'] - df['hip_cx']) / PX_PER_M_X
    df['max_arm_reach_m']= df[['left_reach_m', 'right_reach_m']].max(axis=1)

    # ── Movement Efficiency Index (rolling 2 s window) ────────────────────────
    WINDOW_FR = max(3, int(2.0 * FPS))
    foot_x_np  = df['foot_x_sm'].values
    foot_y_sm_np = savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)

    mei_values = []
    for i in range(len(df)):
        start    = max(0, i - WINDOW_FR)
        xs       = foot_x_np[start:i+1]
        ys       = foot_y_sm_np[start:i+1]
        dx_mei   = np.diff(xs) / PX_PER_M_X
        dy_mei   = np.diff(ys) / PX_PER_M_Y
        path_len = np.sum(np.sqrt(dx_mei**2 + dy_mei**2))
        net_disp = np.sqrt(((xs[-1] - xs[0]) / PX_PER_M_X)**2 +
                           ((ys[-1] - ys[0]) / PX_PER_M_Y)**2)
        mei = net_disp / path_len if path_len > 0.01 else 1.0
        mei_values.append(min(mei, 1.0))
    df['movement_efficiency'] = mei_values

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_cols = [
        'frame_id', 'timestamp',
        'speed_mps', 'accel_mps2',
        'court_zone', 'lateral_zone', 'depth_zone',
        'stride_freq_hz',
        'jump_height_px',
        'lat_reach_m', 'max_arm_reach_m',
        'movement_efficiency',
        'is_burst'
    ]
    df[output_cols].to_csv(output_csv_path, index=False)

    # ── Build frame_entries (mirrors notebook cell) ───────────────────────────
    # Build a fast frame_id → raw frame lookup
    raw_frame_lookup = {fr['frame_id']: fr for fr in frames_raw}

    # Smoothed displacement for direction angle
    dx_sm      = df['foot_x_sm'].diff().fillna(0).values
    dy_sm_vals = pd.Series(
        savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)
    ).diff().fillna(0).values

    frame_entries = []
    for i, row in df.iterrows():
        fid = int(row['frame_id'])

        bb = {
            'x':      round(float(row['bb_x1']), 1),
            'y':      round(float(row['bb_y1']), 1),
            'width':  round(float(row['bb_w']),  1),
            'height': round(float(row['bb_h']),  1),
        }

        src_frame = raw_frame_lookup.get(fid, {})
        raw_kps   = src_frame.get('keypoints', [])
        keypoints = [
            {
                'name':       kp['name'],
                'x':          round(float(kp['x']), 1),
                'y':          round(float(kp['y']), 1),
                'confidence': round(float(kp['confidence']), 3),
            }
            for kp in raw_kps
            if kp.get('confidence', 0) >= CONF_THR
        ]

        court_x = round(float(row['foot_x']) / PX_PER_M_X, 2) \
            if not math.isnan(row['foot_x']) else None
        court_y = round(float(row['foot_y']) / PX_PER_M_Y, 2) \
            if not math.isnan(row['foot_y']) else None

        spd  = round(float(row['speed_mps']),  3) \
            if not math.isnan(row.get('speed_mps',  float('nan'))) else 0.0
        acc  = round(float(row['accel_mps2']), 3) \
            if not math.isnan(row.get('accel_mps2', float('nan'))) else 0.0
        dirn = _movement_direction(dx_sm[i], dy_sm_vals[i])

        lf = {'x': round(float(row['lax']), 1), 'y': round(float(row['lay']), 1)} \
            if not math.isnan(row['lax']) else None
        rf = {'x': round(float(row['rax']), 1), 'y': round(float(row['ray']), 1)} \
            if not math.isnan(row['rax']) else None

        jh   = round(float(row['jump_height_px']), 2)
        step = _classify_step(jh, spd, acc)

        frame_entries.append({
            'frame_id':  fid,
            'timestamp': round(float(row['timestamp']), 4),
            'bounding_box': bb,
            'pose': {'keypoints': keypoints},
            'center_position': {
                'court_x': court_x,
                'court_y': court_y,
            },
            'movement': {
                'speed':        spd,
                'acceleration': acc,
                'direction':    dirn,
            },
            'footwork': {
                'step_type':  step,
                'left_foot':  lf,
                'right_foot': rf,
            },
            'status': {
                'is_moving':    spd > 0.3,
                'is_jumping':   jh > 15,
                'is_recovering': bool(row.get('is_burst', False)),
            },
            'court_zone': str(row['court_zone']),
            'jump_height_px': jh,
        })

    # ── Aggregated metrics (notebook cell logic) ───────────────────────────────
    dx_m_all   = df['foot_x_sm'].diff().fillna(0) / PX_PER_M_X
    dy_m_all   = pd.Series(
        savgol_filter(df['foot_y'].ffill().bfill(), WIN, 2)
    ).diff().fillna(0) / PX_PER_M_Y
    total_distance = float(np.sum(np.sqrt(dx_m_all**2 + dy_m_all**2)))

    avg_speed = float(df['speed_mps'].mean())
    max_speed = float(df['speed_mps'].max())

    # Overall MEI: net A→B displacement / total path length
    net_displacement = float(np.sqrt(
        ((df['foot_x_sm'].iloc[-1] - df['foot_x_sm'].iloc[0]) / PX_PER_M_X)**2 +
        ((df['foot_y'].iloc[-1]    - df['foot_y'].iloc[0])    / PX_PER_M_Y)**2
    ))
    overall_mei = round(min(net_displacement / total_distance, 1.0), 3) \
        if total_distance > 0 else 1.0

    # Court coverage: 10×10 grid, count unique cells
    GRID   = 10
    cx_norm = (df['foot_x'].dropna() / FRAME_W * GRID).astype(int).clip(0, GRID - 1)
    cy_norm = (df['foot_y'].dropna() / FRAME_H * GRID).astype(int).clip(0, GRID - 1)
    cells_visited      = len(set(zip(cx_norm, cy_norm)))
    court_coverage_pct = round(cells_visited / GRID**2 * 100, 1)

    jump_count = int(len(jump_peaks))

    # Average recovery time (40th-percentile speed threshold)
    BURST_END_SPEED = df['speed_mps'].quantile(0.40)
    recovery_times  = []
    speed_vals      = df['speed_mps'].fillna(0).values
    timestamps_arr  = df['timestamp'].values
    for bf in burst_frames:
        for j in range(bf, min(bf + int(FPS * 3), len(speed_vals))):
            if speed_vals[j] < BURST_END_SPEED:
                recovery_times.append(timestamps_arr[j] - timestamps_arr[bf])
                break
    avg_recovery_time = round(float(np.mean(recovery_times)), 3) \
        if recovery_times else 0.0

    # Pose stability: 1 − normalised std of hip_y
    hip_y_std   = df['hip_y'].std()
    hip_y_range = df['hip_y'].max() - df['hip_y'].min()
    pose_stability = round(float(1.0 - min(hip_y_std / (hip_y_range + 1e-6), 1.0)), 3)

    aggregated_metrics = {
        'total_distance_covered':    round(total_distance, 2),
        'average_speed':             round(avg_speed, 3),
        'max_speed':                 round(max_speed, 3),
        'movement_efficiency':       overall_mei,
        'court_coverage_percentage': court_coverage_pct,
        'jump_count':                jump_count,
        'average_recovery_time':     avg_recovery_time,
        'pose_stability_score':      pose_stability,
    }

    # ── Save metrics JSON (includes frame_entries) ────────────────────────────
    output_doc = {
        'match_id':           match_id,
        'player_id':          player_id,
        'frames':             frame_entries,
        'aggregated_metrics': aggregated_metrics,
    }
    with open(output_json_path, 'w') as f:
        json.dump(output_doc, f, indent=2)

    return aggregated_metrics
