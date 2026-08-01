"""Shared video transcoding helper.

OpenCV's VideoWriter produces mp4v-codec files that most browsers can't
play back directly. Module 1's notebook already had a plain ffmpeg
subprocess call for this (cell 3); Module 2's notebook used a second,
independent imageio_ffmpeg-based path for the same job. We keep only this
one so the whole system depends on a single ffmpeg installation (roadmap
finding 0.6 / Step 12).
"""
import subprocess


def transcode_to_h264(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "fast",
        str(output_path),
    ], check=True)
