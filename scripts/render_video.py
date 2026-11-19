#!/usr/bin/env python3
"""
Step 3B: Master video renderer & Director Proof Sheet generator
for "科拉超深钻孔｜通往地狱的井" (1920x1080, 24fps, 7,200 frames exact = 05:00).
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfx_utils import W, H, FPS, BG, AMBER, apply_film_grain, get_font
from shots_act1_2 import (
    render_shot01, render_shot02, render_shot03, render_shot04, render_shot05,
    render_shot06, render_shot07, render_shot08, render_shot09,
)
from shots_act3_4 import (
    render_shot10, render_shot11, render_shot12, render_shot13, render_shot14,
    render_shot15, render_shot16, render_shot17, render_shot18,
)
from shots_act5 import (
    render_shot19, render_shot20, render_shot21, render_shot22,
    render_shot23, render_shot24, render_shot25, render_shot26,
)

ROOT = "/home/antigravity/kola_superdeep"
GRADED_DIR = f"{ROOT}/assets/graded"
AUDIO_DIR = f"{ROOT}/audio"
RENDER_DIR = f"{ROOT}/renders"
OUT_DIR = f"{ROOT}/output"
os.makedirs(RENDER_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Exact 26-shot table matching kola_superdeep_script.md (total = 300s = 7,200 frames @ 24fps)
SHOTS = [
    (1,   0,   3, render_shot01, 5.5),
    (2,   3,  15, render_shot02, 7.0),
    (3,  15,  28, render_shot03, 5.0),
    (4,  28,  36, render_shot04, 7.5),
    (5,  36,  40, render_shot05, 8.5),   # Part 3.1: 12% film grain on title
    (6,  40,  55, render_shot06, 7.5),
    (7,  55,  66, render_shot07, 4.5),
    (8,  66,  82, render_shot08, 6.5),
    (9,  82,  95, render_shot09, 7.5),
    (10, 95, 106, render_shot10, 4.5),
    (11, 106, 126, render_shot11, 4.5),
    (12, 126, 141, render_shot12, 6.5),
    (13, 141, 155, render_shot13, 6.5),
    (14, 155, 162, render_shot14, 7.0),
    (15, 162, 176, render_shot15, 4.5),
    (16, 176, 189, render_shot16, 6.0),
    (17, 189, 199, render_shot17, 7.0),
    (18, 199, 215, render_shot18, 5.5),
    (19, 215, 227, render_shot19, 4.5),
    (20, 227, 240, render_shot20, 4.5),
    (21, 240, 252, render_shot21, 6.5),
    (22, 252, 260, render_shot22, 7.5),
    (23, 260, 271, render_shot23, 7.0),
    (24, 271, 284, render_shot24, 4.5),
    (25, 284, 294, render_shot25, 5.5),
    (26, 294, 300, render_shot26, 5.0),
]

def load_ctx():
    cues = json.load(open(f"{AUDIO_DIR}/sync_cues.json"))
    hell_tape = np.load(f"{AUDIO_DIR}/hell_tape_mono.npy")
    plates = {}
    for fname in os.listdir(GRADED_DIR):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            key = os.path.splitext(fname)[0]
            plates[key] = Image.open(f"{GRADED_DIR}/{fname}").convert("RGB")
    return {"cues": cues, "hell_tape": hell_tape, "plates": plates}

def build_proof_sheets():
    ctx = load_ctx()
    # Representative moments covering all 26 shots + dual-phase shots
    samples = [
        ("S01 00:02.4 Waveform Clip", 1, 2.4),
        ("S02 00:10.0 CRT + HUD", 2, 7.0),
        ("S03 00:25.5 1100 vs 180 + FFT", 3, 10.5),
        ("S04 00:32.0 Blizzard Ruin", 4, 4.0),
        ("S05 00:38.2 Opening Title", 5, 2.2),
        ("S06a 00:45.0 Kola Tower", 6, 5.0),
        ("S06b 00:51.0 Moon Insert", 6, 11.0),
        ("S07 01:04.2 -15,000 m", 7, 9.2),
        ("S08a 01:12.5 Moho Ray", 8, 6.5),
        ("S08b 01:20.5 Mohole Cancelled", 8, 14.5),
        ("S09a 01:25.5 Rig Tongs", 9, 3.5),
        ("S09b 01:31.5 Rotary Hole", 9, 9.5),
        ("S10 01:39.5 Hardness Curve", 10, 4.5),
        ("S11 02:00.0 180°C Divergence", 11, 14.0),
        ("S12 02:15.0 Plastic Granite", 12, 9.0),
        ("S13 02:29.0 Closing Wall", 13, 8.0),
        ("S14 02:38.5 Core Archive", 14, 3.5),
        ("S15 02:52.0 01 No Basalt", 15, 10.0),
        ("S16a 03:00.0 02 3D Lattice", 16, 4.0),
        ("S16b 03:06.5 02 Deep Water", 16, 10.5),
        ("S17 03:16.0 03 H2 Mud", 17, 7.0),
        ("S18 03:29.0 04 Microfossil", 18, 10.0),
        ("S19 03:42.5 5000m Lost", 19, 7.5),
        ("S20 03:58.0 12,262 m Lock", 20, 11.0),
        ("S21a 04:04.2 300°C Extrap", 21, 4.2),
        ("S21b 04:08.5 Control Room", 21, 8.5),
        ("S22 04:16.0 Ruin Snow", 22, 4.0),
        ("S23 04:27.0 23cm Cap", 23, 7.0),
        ("S24a 04:38.0 0.19% Earth", 24, 7.0),
        ("S24b 04:42.5 Apple Skin", 24, 11.5),
        ("S25 04:51.5 Polar Stars", 25, 7.5),
        ("S26 04:55.5 End Card", 26, 1.5),
    ]
    shot_map = {s[0]: s for s in SHOTS}

    def save_sheet(batch, out_path):
        cols = 4
        tw, th = 480, 270
        pad_top = 28
        rows = (len(batch) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * (th + pad_top)), (14, 14, 18))
        sd = ImageDraw.Draw(sheet)
        for i, (label, snum, tl) in enumerate(batch):
            _, st, en, fn, gr = shot_map[snum]
            im = fn(tl, float(en - st), ctx)
            arr = apply_film_grain(im, int((st + tl) * FPS), strength=gr)
            thumb = Image.fromarray(arr).resize((tw - 6, th - 6), Image.Resampling.LANCZOS)
            r, c = divmod(i, cols)
            sheet.paste(thumb, (c * tw + 3, r * (th + pad_top) + pad_top + 3))
            sd.text((c * tw + 8, r * (th + pad_top) + 5), label, font=get_font("mono_b", 15), fill=AMBER)
        sheet.save(out_path, quality=90)
        print("Saved proof sheet:", out_path)

    save_sheet(samples[:16], f"{ROOT}/assets/proof_sheet_1.jpg")
    save_sheet(samples[16:], f"{ROOT}/assets/proof_sheet_2.jpg")

def render_single_shot(shot_tuple):
    snum, st, en, fn, gr = shot_tuple
    dur = float(en - st)
    n_frames = int(round(dur * FPS))
    out_mp4 = f"{RENDER_DIR}/shot_{snum:02d}.mp4"
    t0 = time.time()
    ctx = load_ctx()

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15", "-threads", "1", "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for fi in range(n_frames):
        tl = fi / FPS
        im = fn(tl, dur, ctx)
        # Skip grain on pure-black trailing silence of Shot 26 (>= 3.0s) for cleanest fade-to-black
        if snum == 26 and tl >= 3.0:
            arr = np.full((H, W, 3), BG, dtype=np.uint8)
        else:
            arr = apply_film_grain(im, st * FPS + fi, strength=gr)
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="ignore")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg failed on shot {snum}: {err[-500:]}")
    dt = time.time() - t0
    print(f"[Shot {snum:02d}] {n_frames:3d}f ({dur:4.1f}s) rendered in {dt:5.1f}s ({n_frames/dt:4.1f} fps)", flush=True)
    return out_mp4

def assemble_master():
    concat_txt = f"{RENDER_DIR}/concat.txt"
    with open(concat_txt, "w") as f:
        for s in SHOTS:
            f.write(f"file '{RENDER_DIR}/shot_{s[0]:02d}.mp4'\n")

    sub_ass = f"{ROOT}/subtitles/kola_subtitles.ass"
    audio_wav = f"{AUDIO_DIR}/audio_master.wav"
    final_mp4 = f"{OUT_DIR}/kola_superdeep_1080p.mp4"

    print("\nAssembling final 1080p master with burned-in Part 1.2 subtitles + 48kHz mastered audio...", flush=True)
    # Burn crisp styled ASS subtitles & mux mastered audio cleanly to exact 300.000s
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-i", audio_wav,
        "-vf", f"ass={sub_ass}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-maxrate", "14M", "-bufsize", "28M", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
        "-t", "300.000",
        "-movflags", "+faststart",
        final_mp4,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Final mux failed:\n{res.stderr[-1000:]}")
    print(f"SUCCESS! Final master video ready -> {final_mp4}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["proof", "render", "mux"], default="proof")
    ap.add_argument("--shots", type=str, default="", help="Optional comma-separated shot numbers or ranges (e.g. 4-15)")
    args = ap.parse_args()

    if args.mode == "proof":
        build_proof_sheets()
    elif args.mode == "render":
        target = SHOTS
        if args.shots:
            ids = set()
            for tok in args.shots.split(","):
                tok = tok.strip()
                if "-" in tok:
                    a, b = tok.split("-", 1)
                    ids.update(range(int(a), int(b) + 1))
                elif tok:
                    ids.add(int(tok))
            target = [s for s in SHOTS if s[0] in ids]
        t0 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
            list(ex.map(render_single_shot, target))
        print(f"All {len(target)} shot clips rendered in {time.time() - t0:.1f}s!", flush=True)
        if not args.shots:
            assemble_master()
    elif args.mode == "mux":
        assemble_master()

if __name__ == "__main__":
    main()
