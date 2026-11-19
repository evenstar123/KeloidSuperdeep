#!/usr/bin/env python3
"""
Step 1: Generate precision-timed Mandarin documentary narration (Kokoro v1.1-zh),
exact phrase/subtitle timeline JSON, and styled ASS subtitles (Part 1.2 spec).
"""
import json
import os
import re
import subprocess
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import sherpa_onnx

SCRIPT_PATH = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/8660d99d-b4d4-453e-9341-bb8360253168/kola_superdeep_script.md"
ROOT = "/home/antigravity/kola_superdeep"
MODEL_DIR = f"{ROOT}/models/kokoro-int8-multi-lang-v1_1"
AUDIO_DIR = f"{ROOT}/audio"
SUB_DIR = f"{ROOT}/subtitles"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

SID = 90  # Deep, restrained Mandarin male documentary baritone
TARGET_SR = 48000
TOTAL_SEC = 300.0

# Subtle TTS-only pronunciation / segmentation helpers (subtitles keep exact script text)
TTS_FIXES = [
    ("〇", "零"),
    ("温度计停在了", "温度计，停在了"),
    ("井，是真的。", "井，是真的。"),
    ("钻头一撤出", "钻头，一撤出"),
    ("打捞失败。", "打捞失败。"),
]

# Optional clean display formatting for subtitles (<= 18 chars per line)
def clean_tts_text(t: str) -> str:
    for a, b in TTS_FIXES:
        t = t.replace(a, b)
    return t.strip()

def split_sub_chunks(sentence: str, max_len: int = 18):
    """Split a spoken sentence into readable <=18-char subtitle chunks."""
    s = sentence.strip()
    # Strip trailing period for cleaner Bilibili/LEMMiNO documentary subtitles, keep internal punctuation
    disp = s.rstrip("。")
    if len(disp) <= max_len:
        return [disp]
    # Split on Chinese commas / colons / semicolons / enumeration commas
    parts = re.split(r"([，：；、])", disp)
    chunks = []
    cur = ""
    i = 0
    while i < len(parts):
        piece = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        seg = piece + punct
        if cur and len(cur) + len(seg.rstrip("，：；、")) > max_len:
            chunks.append(cur.rstrip("，：；、"))
            cur = seg
        else:
            cur += seg
        i += 2
    if cur:
        chunks.append(cur.rstrip("，：；、"))
    # Safety wrap if any single chunk still > max_len
    final = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            mid = len(c) // 2
            final.append(c[:mid])
            final.append(c[mid:])
    return [c for c in final if c]

def parse_script():
    txt = open(SCRIPT_PATH, encoding="utf-8").read()
    shots = []
    pattern = re.compile(
        r"### \[Shot (\d+) \| (\d\d):(\d\d) - (\d\d):(\d\d) \| 资产类型：([^\]]+)\]\n"
        r"- \*\*旁白文案\*\*：([^\n]+)"
    )
    for m in pattern.finditer(txt):
        num = int(m.group(1))
        st = int(m.group(2)) * 60 + int(m.group(3))
        en = int(m.group(4)) * 60 + int(m.group(5))
        asset_type = m.group(6).strip()
        nar = m.group(7).strip()
        shots.append({
            "shot": num,
            "start": float(st),
            "end": float(en),
            "dur": float(en - st),
            "asset_type": asset_type,
            "raw_narration": nar,
        })
    assert len(shots) == 26 and shots[-1]["end"] == TOTAL_SEC, f"Unexpected shots: {len(shots)}"
    return shots

def tokenize_narration(nar: str):
    """Return ordered list of ('pause', sec) or ('speech', sentence_str)."""
    if nar == "[纯画面留白]":
        return []
    tokens = []
    parts = re.split(r"(\[(?:停顿|留白) [\d.]+s\])", nar)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        pm = re.match(r"\[(停顿|留白) ([\d.]+)s\]", p)
        if pm:
            tokens.append(("pause", float(pm.group(2)), pm.group(1)))
        else:
            # Split into sentences by 。 so we get tight subtitle timing & visual cue anchors
            sents = [x.strip() for x in re.split(r"(?<=。)", p) if x.strip()]
            for s in sents:
                tokens.append(("speech", s, "text"))
    return tokens

def trim_silence(arr: np.ndarray, sr: int, thresh: float = 0.008, pad_ms: int = 35):
    idx = np.where(np.abs(arr) > thresh)[0]
    if len(idx) == 0:
        return arr
    pad = int(sr * pad_ms / 1000)
    a = max(0, idx[0] - pad)
    b = min(len(arr), idx[-1] + pad)
    out = arr[a:b].copy()
    # 12ms smooth fade in/out to avoid any click
    f = min(int(sr * 0.012), len(out) // 4)
    if f > 0:
        out[:f] *= np.linspace(0, 1, f, dtype=np.float32)
        out[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return out

def resample_to_48k(arr: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return arr
    return signal.resample_poly(arr, TARGET_SR, src_sr).astype(np.float32)

def fmt_ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def main():
    print("Loading Kokoro v1.1-zh TTS model...")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{MODEL_DIR}/model.int8.onnx",
                voices=f"{MODEL_DIR}/voices.bin",
                tokens=f"{MODEL_DIR}/tokens.txt",
                data_dir=f"{MODEL_DIR}/espeak-ng-data",
                dict_dir=f"{MODEL_DIR}/dict",
                lexicon=f"{MODEL_DIR}/lexicon-us-en.txt,{MODEL_DIR}/lexicon-zh-custom.txt",
            ),
            num_threads=2,
            provider="cpu",
        ),
        rule_fsts=f"{MODEL_DIR}/date-zh.fst,{MODEL_DIR}/number-zh.fst,{MODEL_DIR}/phone-zh.fst",
    )
    tts = sherpa_onnx.OfflineTts(cfg)
    src_sr = tts.sample_rate
    shots = parse_script()

    master = np.zeros(int(TOTAL_SEC * TARGET_SR), dtype=np.float32)
    timeline = {"total_sec": TOTAL_SEC, "sample_rate": TARGET_SR, "shots": [], "subtitles": []}

    for sh in shots:
        tokens = tokenize_narration(sh["raw_narration"])
        shot_rec = {
            "shot": sh["shot"],
            "start": sh["start"],
            "end": sh["end"],
            "dur": sh["dur"],
            "events": [],
        }
        if not tokens:
            timeline["shots"].append(shot_rec)
            print(f"Shot {sh['shot']:02d} [{sh['start']:5.1f}-{sh['end']:5.1f}] PURE SILENCE ({sh['dur']:.1f}s)")
            continue

        # Synthesize at documentary speed=1.08 first, check fit inside shot window
        base_speed = 1.08
        inter_sent_gap = 0.22  # crisp natural breath between sentences

        def synth_all(spd):
            segs = []
            for kind, val, tag in tokens:
                if kind == "pause":
                    segs.append(("pause", val, tag, None))
                else:
                    a = tts.generate(clean_tts_text(val), sid=SID, speed=spd)
                    arr = trim_silence(np.array(a.samples, dtype=np.float32), src_sr)
                    arr48 = resample_to_48k(arr, src_sr)
                    segs.append(("speech", val, tag, arr48))
            return segs

        segs = synth_all(base_speed)

        def calc_total(segs_list, gap):
            t = 0.0
            prev_speech = False
            for kind, val, tag, arr in segs_list:
                if kind == "pause":
                    t += val
                    prev_speech = False
                else:
                    if prev_speech:
                        t += gap
                    t += len(arr) / TARGET_SR
                    prev_speech = True
            return t

        total_need = calc_total(segs, inter_sent_gap)
        avail = sh["dur"] - 0.15  # leave clean breath margin inside shot cut

        # If speech + pauses exceeds shot duration, compute exact clean speed adjustment
        if total_need > avail:
            pause_sum = sum(v for k, v, _, _ in segs if k == "pause")
            # Keep trailing [留白] intact if possible; gently scale speed up
            speech_sum = total_need - pause_sum
            target_speech = max(avail - pause_sum, speech_sum * 0.80)
            new_speed = min(1.30, base_speed * (speech_sum / target_speech) * 1.02)
            segs = synth_all(new_speed)
            total_need = calc_total(segs, 0.16)
            inter_sent_gap = 0.16
            # If STILL slightly over (only possible if trailing silence + speech barely overlaps), trim trailing pause slightly
            if total_need > avail and segs[-1][0] == "pause":
                overflow = total_need - avail
                k, v, tg, a = segs[-1]
                segs[-1] = (k, max(0.8, v - overflow), tg, a)
                total_need = calc_total(segs, inter_sent_gap)
            # Final safety: time-stretch audio gently if still > avail
            if total_need > avail:
                ratio = avail / total_need
                inter_sent_gap *= ratio
                new_segs = []
                for k, v, tg, arr in segs:
                    if k == "pause":
                        new_segs.append((k, v * ratio, tg, None))
                    else:
                        new_len = int(len(arr) * ratio)
                        new_segs.append((k, v, tg, signal.resample(arr, new_len).astype(np.float32)))
                segs = new_segs
                total_need = calc_total(segs, inter_sent_gap)

        # Place inside shot: if first token isn't a pause, start 0.12s after cut for cinematic polish
        lead = 0.0 if segs[0][0] == "pause" else min(0.14, max(0.0, (sh["dur"] - total_need) * 0.25))
        cursor = sh["start"] + lead
        prev_speech = False

        for kind, val, tag, arr in segs:
            if kind == "pause":
                shot_rec["events"].append({
                    "type": "pause",
                    "tag": tag,
                    "start": round(cursor, 3),
                    "end": round(cursor + val, 3),
                })
                cursor += val
                prev_speech = False
            else:
                if prev_speech:
                    cursor += inter_sent_gap
                st_t = cursor
                dur_s = len(arr) / TARGET_SR
                en_t = min(sh["end"] - 0.02, st_t + dur_s)
                i0 = int(round(st_t * TARGET_SR))
                i1 = min(len(master), i0 + len(arr))
                n_copy = min(i1 - i0, int(round((en_t - st_t) * TARGET_SR)))
                if n_copy > 0:
                    master[i0:i0 + n_copy] += arr[:n_copy]
                shot_rec["events"].append({
                    "type": "speech",
                    "text": val,
                    "start": round(st_t, 3),
                    "end": round(en_t, 3),
                })
                # Generate <=18-char subtitles proportional to character count
                sub_chunks = split_sub_chunks(val, 18)
                total_chars = sum(max(1, len(c)) for c in sub_chunks)
                sub_cur = st_t
                for c in sub_chunks:
                    c_dur = (en_t - st_t) * (max(1, len(c)) / total_chars)
                    timeline["subtitles"].append({
                        "shot": sh["shot"],
                        "start": round(sub_cur, 3),
                        "end": round(sub_cur + c_dur, 3),
                        "text": c,
                    })
                    sub_cur += c_dur
                cursor = en_t
                prev_speech = True

        timeline["shots"].append(shot_rec)
        speech_s = sum(e["end"] - e["start"] for e in shot_rec["events"] if e["type"] == "speech")
        print(f"Shot {sh['shot']:02d} [{sh['start']:5.1f}-{sh['end']:5.1f}] dur={sh['dur']:4.1f}s speech={speech_s:4.1f}s end_at={cursor - sh['start']:4.1f}s")

    # Normalize raw narration & export temp mono wav, then master via ffmpeg vocal chain
    peak = np.max(np.abs(master)) + 1e-9
    master = (master / peak) * 0.85
    raw_wav = f"{AUDIO_DIR}/narration_raw.wav"
    wav.write(raw_wav, TARGET_SR, (np.clip(master, -1, 1) * 32767).astype(np.int16))

    # Broadcast documentary voice mastering:
    # HPF 75Hz, chest warmth +2dB @ 155Hz, clear presence +1.8dB @ 3.2kHz, gentle de-harsh -1.5dB @ 6.5kHz,
    # smooth 2.5:1 compressor, subtle dark room impulse echo, clean stereo output
    master_wav = f"{AUDIO_DIR}/narration_master.wav"
    af = (
        "highpass=f=75,"
        "equalizer=f=155:width_type=o:width=1.0:g=2.0,"
        "equalizer=f=3200:width_type=o:width=1.2:g=1.8,"
        "equalizer=f=6800:width_type=o:width=1.0:g=-1.5,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=15:release=140:makeup=2,"
        "aecho=0.92:0.88:28:0.06,"
        "alimiter=limit=-1.5dB,"
        "pan=stereo|c0=c0|c1=c0"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_wav, "-af", af,
        "-ar", str(TARGET_SR), "-ac", "2", "-t", f"{TOTAL_SEC:.3f}",
        master_wav
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Save timeline JSON
    with open(f"{AUDIO_DIR}/narration_timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    # Write styled ASS subtitles (Part 1.2: Noto Sans CJK SC, #D9DDE0 @ 90% opacity, bottom-centered)
    # ASS color format: &HAABBGGRR (alpha 0x1A ≈ 90% opaque; RGB D9DDE0 -> BGR E0DDD9)
    ass_header = """[Script Info]
Title: 科拉超深钻孔｜通往地狱的井
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: DocSub,Noto Sans CJK SC,42,&H1AE0DDD9,&H1AE0DDD9,&H080A0A0C,&H400A0A0C,0,0,0,0,100,100,2.5,0,1,2.8,2.2,2,120,120,54,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [ass_header]
    for sub in timeline["subtitles"]:
        st = fmt_ass_time(sub["start"])
        en = fmt_ass_time(sub["end"])
        txt = sub["text"].replace("\n", "\\N")
        # Gentle 80ms fade in/out for cinema documentary subtitles
        lines.append(f"Dialogue: 0,{st},{en},DocSub,,0,0,0,,{{\\fad(80,80)}}{txt}\n")
    with open(f"{SUB_DIR}/kola_subtitles.ass", "w", encoding="utf-8") as f:
        f.writelines(lines)

    tot_sp = sum(
        e["end"] - e["start"]
        for sh in timeline["shots"]
        for e in sh["events"]
        if e["type"] == "speech"
    )
    print(f"\nDONE! Master narration: {master_wav}")
    print(f"Total active narration duration: {tot_sp:.1f}s ({tot_sp/3:.1f}%), Silence/breathing: {300-tot_sp:.1f}s")
    print(f"Total subtitle events: {len(timeline['subtitles'])}")

if __name__ == "__main__":
    main()
