"""
Module 1 — annotated video rendering (pose overlay + movement overlay).
Content moved verbatim out of the original api-ready notebook's cells 2
(COL / SKELETON_PAIRS), 5 (draw helpers), 7 (render_pose_video), and 9
(render_movement_video).
"""
import math

import cv2
import numpy as np
import json

from .tracking import draw_pose

# ------------------------------------------------------------------
# Colour palette + skeleton pairs (notebook cell 2)
# ------------------------------------------------------------------
COL = {
    'tracked':    (0, 220, 0),
    'predicted':  (0, 165, 255),
    'jump':       (0, 0, 255),
    'sprint':     (255, 100, 0),
    'lunge':      (180, 0, 255),
    'step':       (0, 200, 200),
    'stand':      (180, 180, 180),
    'skeleton':   (0, 255, 255),
    'kp_dot':     (0, 0, 255),
    'text_bg':    (20, 20, 20),
    'text_white': (255, 255, 255),
    'speed_bar':  (0, 200, 100),
    'accel_pos':  (0, 180, 255),
    'accel_neg':  (0, 60, 255),
}

SKELETON_PAIRS = [
    ('left_shoulder', 'left_elbow'),   ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),     ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),         ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),       ('right_knee', 'right_ankle'),
    ('nose', 'left_eye'),              ('nose', 'right_eye'),
]


# ------------------------------------------------------------------
# Draw helpers (notebook cell 5)
# ------------------------------------------------------------------
def _draw_skeleton(frame, keypoints):
    kp_map = {kp['name']: (int(kp['x']), int(kp['y'])) for kp in keypoints}
    for a, b in SKELETON_PAIRS:
        if a in kp_map and b in kp_map:
            cv2.line(frame, kp_map[a], kp_map[b], COL['skeleton'], 2)
    for pt in kp_map.values():
        cv2.circle(frame, pt, 4, COL['kp_dot'], -1)


def _draw_text_box(frame, lines, origin, font_scale=0.5, thickness=1, padding=5):
    """Dark semi-transparent box then white text."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(font_scale * 28)
    max_w  = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    x0, y0 = origin
    box_h  = line_h * len(lines) + padding * 2
    box_w  = max_w + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), COL['text_bg'], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        y = y0 + padding + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x0 + padding, y), font, font_scale,
                    COL['text_white'], thickness, cv2.LINE_AA)


def _draw_speed_bar(frame, speed, max_spd=10.0, origin=(20, 80), bar_w=180, bar_h=14):
    x, y = origin
    fill = int(min(speed / max_spd, 1.0) * bar_w)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + fill,  y + bar_h), COL['speed_bar'], -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f'{speed:.1f} m/s', (x + bar_w + 6, y + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL['text_white'], 1, cv2.LINE_AA)


def _draw_direction_arrow(frame, cx, cy, direction_deg, speed, length=40):
    if speed < 0.3:
        return
    rad = math.radians(direction_deg)
    ex  = int(cx + length * math.cos(rad))
    ey  = int(cy - length * math.sin(rad))   # image Y inverted
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (255, 220, 0), 2, tipLength=0.3)


# ------------------------------------------------------------------
# Pose overlay video (notebook cell 7)
# ------------------------------------------------------------------
def render_pose_video(video_path, out_mp4, json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames_data = {f['frame_id']: f for f in data['frames']}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    for frame_id in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        fd = frames_data.get(frame_id)
        if fd and fd.get("player_detected"):
            status = fd.get("tracking_status", "missing")
            box_color = (0, 255, 0) if status == "tracked" else (0, 165, 255)
            bb = fd["bounding_box"]
            if bb is not None:
                x1, y1, x2, y2 = int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, f"Target Player | {status}", (x1, max(30, y1 - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)
                pose_name = fd.get("shot_classification", "Detecting...")
                shot_conf = fd.get("shot_confidence", 0.0)
                cv2.putText(frame, f"Shot: {pose_name} ({shot_conf:.2f})", (x1, max(55, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Draw pose
            kpts = np.zeros((17, 2))
            kpt_conf = np.zeros(17)
            for kp in fd.get("keypoints", []):
                idx = kp["id"]
                kpts[idx] = [kp["x"], kp["y"]]
                kpt_conf[idx] = kp["confidence"]
            draw_pose(frame, kpts, kpt_conf)

        out.write(frame)

    cap.release()
    out.release()


# ------------------------------------------------------------------
# Movement overlay video (notebook cell 9)
# ------------------------------------------------------------------
def render_movement_video(json_path, original_video_path, output_video_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    frame_entries = data.get('frames', [])

    if not output_video_path or not original_video_path:
        return

    frame_lookup = {e['frame_id']: e for e in frame_entries}
    jh_lookup = {e['frame_id']: float(e.get('jump_height_px', 0.0)) for e in frame_entries}

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        return

    fw_v    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_v    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_vid = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_out, (fw_v, fh_v)
    )

    for video_frame_id in range(total_fr):
        ret, frame = cap.read()
        if not ret:
            break

        entry = frame_lookup.get(video_frame_id)
        if entry is None:
            out_vid.write(frame)
            continue

        bb  = entry.get('bounding_box', {})
        mv  = entry.get('movement', {})
        fw_ = entry.get('footwork', {})
        st  = entry.get('status', {})
        cp  = entry.get('center_position', {})
        pose = entry.get('pose', {})
        kps = pose.get('keypoints', [])

        x1 = int(bb.get('x', 0))
        y1 = int(bb.get('y', 0))
        x2 = int(x1 + bb.get('width', 0))
        y2 = int(y1 + bb.get('height', 0))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        step_col = COL.get(fw_.get('step_type', 'stand'), COL['stand'])

        cv2.rectangle(frame, (x1, y1), (x2, y2), step_col, 2)

        if kps:
            _draw_skeleton(frame, kps)

        _draw_direction_arrow(frame, cx, cy, mv.get('direction', 0), mv.get('speed', 0))

        zone_label = f"Zone: {entry.get('court_zone', '?')}"
        flags = []
        if st.get('is_jumping'):    flags.append('JUMP')
        if st.get('is_recovering'): flags.append('RECOVER')
        if not flags and st.get('is_moving'):
            flags.append(str(fw_.get('step_type', '')).upper())
        if not flags:
            flags.append('STAND')

        cp_x = cp.get('court_x') if cp.get('court_x') is not None else '?'
        cp_y = cp.get('court_y') if cp.get('court_y') is not None else '?'

        lines = [
            f"Spd: {mv.get('speed', 0):.1f} m/s  Acc: {mv.get('acceleration', 0):+.1f}",
            f"Dir: {mv.get('direction', 0):.0f}deg  Step: {fw_.get('step_type', '')}",
            f"Court: ({cp_x}, {cp_y}) m",
            zone_label,
            '  '.join(flags),
        ]
        _draw_text_box(frame, lines, (max(0, x1), max(0, y1 - 110)))

        _draw_speed_bar(frame, mv.get('speed', 0), origin=(16, 20))

        for foot_key in ('left_foot', 'right_foot'):
            ft = fw_.get(foot_key)
            if ft:
                cv2.circle(frame, (int(ft['x']), int(ft['y'])), 6, (0, 255, 180), -1)

        jh_px = jh_lookup.get(video_frame_id, 0.0)
        if jh_px > 10:
            jh_int   = int(jh_px)
            jh_bar_x = x2 + 6
            cv2.line(frame, (jh_bar_x, y2), (jh_bar_x, max(y2 - jh_int, 0)), (0, 0, 255), 4)
            cv2.putText(frame, f'J:{jh_int}px', (jh_bar_x + 4, max(y2 - jh_int - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

        out_vid.write(frame)

    cap.release()
    out_vid.release()
