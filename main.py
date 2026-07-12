"""
Podcast -> single vertical reel, end to end.

Usage:
    python main.py podcast.mp4 -o reel.mp4
    python main.py podcast.mp4 --llm-model qwen2.5:7b-instruct --target-duration 30
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

from transcribe import transcribe
from planner import plan_edit
from silence import find_silence_gaps, remap_time
from crop import detect_face_center_x, compute_crop_x
from captions import build_ass
from render import cut_and_crop, burn_captions


def get_video_dims(path: str):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    d = json.loads(out)["streams"][0]
    return int(d["width"]), int(d["height"])


def get_video_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def main():
    ap = argparse.ArgumentParser(description="Podcast -> vertical reel AI agent")
    ap.add_argument("input", help="Input podcast video file")
    ap.add_argument("-o", "--output", default="reel_output.mp4")
    ap.add_argument("--whisper-model", default="base",
                     help="tiny/base/small/medium/large-v3 (default: base)")
    ap.add_argument("--llm-model", default="llama3.1:8b", help="Ollama model tag")
    ap.add_argument("--target-duration", type=float, default=45.0)
    ap.add_argument("--silence-gap", type=float, default=0.5,
                     help="Cut internal gaps longer than this many seconds")
    ap.add_argument("--words-per-caption", type=int, default=4)
    ap.add_argument("--keep-work-dir", action="store_true",
                     help="Keep transcript.json / intermediate files for debugging")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    workdir = "work_" + os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(workdir, exist_ok=True)

    print("[1/6] Transcribing (this can take a while on CPU)...")
    t = transcribe(args.input, model_size=args.whisper_model)
    with open(f"{workdir}/transcript.json", "w") as f:
        json.dump(t, f, indent=2)

    print("[2/6] Asking local LLM to pick the best segment...")
    duration = get_video_duration(args.input)
    plan = plan_edit(t["segments"], target_duration=args.target_duration, model=args.llm_model)
    plan["start"] = max(0.0, plan["start"])
    plan["end"] = min(duration, plan["end"])
    print(f"  -> {plan['start']:.1f}s - {plan['end']:.1f}s | {plan.get('reason', '')}")

    print("[3/6] Finding silence gaps to trim...")
    keep_intervals = find_silence_gaps(t["words"], plan["start"], plan["end"],
                                        min_gap=args.silence_gap)

    print("[4/6] Detecting speaker position for crop...")
    frame_w, frame_h = get_video_dims(args.input)
    crop_width = int(frame_h * 9 / 16)
    sample_times = [s + (e - s) / 2 for s, e in keep_intervals[:5]] or [plan["start"]]
    center_x = detect_face_center_x(args.input, sample_times, frame_w)
    crop_x = compute_crop_x(center_x, crop_width, frame_w)

    print("[5/6] Cutting silence + cropping to 9:16...")
    cut_path = f"{workdir}/cut.mp4"
    cut_and_crop(args.input, keep_intervals, crop_x, crop_width, frame_h, cut_path)

    print("      Building word-synced captions...")
    words_in_seg = [w for w in t["words"] if w["start"] >= plan["start"] and w["end"] <= plan["end"]]
    remapped_words = [
        {"start": remap_time(w["start"], keep_intervals),
         "end": remap_time(w["end"], keep_intervals),
         "word": w["word"]}
        for w in words_in_seg
    ]
    ass_path = f"{workdir}/captions.ass"
    build_ass(remapped_words, ass_path, words_per_line=args.words_per_caption)

    print("[6/6] Burning captions into final render...")
    burn_captions(cut_path, ass_path, args.output)

    if not args.keep_work_dir:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"  (intermediate files kept in {workdir}/)")

    print(f"\nDone -> {args.output}")


if __name__ == "__main__":
    main()
