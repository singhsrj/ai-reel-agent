"""
Two-pass ffmpeg rendering:
  Pass 1: select/aselect out the silence gaps, crop to 9:16 around the
          speaker, scale to 1080x1920.
  Pass 2: burn the .ass captions onto the result.
Two passes trade a bit of speed for being much easier to debug than one
giant filtergraph.
"""
import subprocess


def _select_expr(keep_intervals: list) -> str:
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keep_intervals)


def cut_and_crop(video_path: str, keep_intervals: list, crop_x: int, crop_width: int,
                  crop_height: int, out_path: str) -> None:
    expr = _select_expr(keep_intervals)
    vf = (
        f"select='{expr}',setpts=N/FRAME_RATE/TB,"
        f"crop={crop_width}:{crop_height}:{crop_x}:0,"
        f"scale=1080:1920"
    )
    af = f"aselect='{expr}',asetpts=N/SR/TB"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf, "-af", af,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def burn_captions(video_path: str, ass_path: str, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True)
