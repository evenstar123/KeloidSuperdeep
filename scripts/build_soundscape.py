#!/usr/bin/env python3
"""
Step 2: Cinema-grade 3-layer sound design (Ambience + Foley + Score) + final master mix
for "Kola Superdeep Borehole: The Well to Hell" (05:00 exact, 48kHz stereo).
Also saves exact Shot 01-03 tape waveform envelope & spectrogram data for audio-reactive MG visuals.
"""
import json
import os
import subprocess
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

ROOT = "/home/antigravity/kola_superdeep"
AUDIO_DIR = f"{ROOT}/audio"
SR = 48000
TOTAL_SEC = 300.0
N = int(TOTAL_SEC * SR)
RNG = np.random.default_rng(19700524)  # Kola drilling start date seed

def sec_slice(t0: float, t1: float):
    i0 = max(0, int(round(t0 * SR)))
    i1 = min(N, int(round(t1 * SR)))
    return i0, i1, max(0, i1 - i0)

def env_ad(n: int, att_s: float, dec_s: float):
    a = min(n, max(1, int(att_s * SR)))
    d = min(n - a, max(1, int(dec_s * SR)))
    e = np.zeros(n, dtype=np.float32)
    e[:a] = np.linspace(0, 1, a, dtype=np.float32) ** 1.5
    e[a:a + d] = np.linspace(1, 0, d, dtype=np.float32) ** 2.2
    return e

def env_asr(n: int, att_s: float, rel_s: float):
    a = min(n // 2, max(1, int(att_s * SR)))
    r = min(n // 2, max(1, int(rel_s * SR)))
    e = np.ones(n, dtype=np.float32)
    e[:a] = np.sin(np.linspace(0, np.pi / 2, a, dtype=np.float32))
    e[-r:] = np.sin(np.linspace(np.pi / 2, 0, r, dtype=np.float32))
    return e

def pink_noise(n: int) -> np.ndarray:
    """Fast Voss-McCartney / FFT 1/f pink noise."""
    w = RNG.standard_normal(n).astype(np.float32)
    f = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    freqs[0] = 1.0
    f /= np.sqrt(freqs)
    y = np.fft.irfft(f, n=n).astype(np.float32)
    return y / (np.std(y) + 1e-9)

def bp_filter(x: np.ndarray, low: float, high: float, order: int = 3):
    sos = signal.butter(order, [max(10.0, low), min(SR * 0.45, high)], btype="bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)

def lp_filter(x: np.ndarray, cutoff: float, order: int = 3):
    sos = signal.butter(order, min(SR * 0.45, max(20.0, cutoff)), btype="lowpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)

def hp_filter(x: np.ndarray, cutoff: float, order: int = 3):
    sos = signal.butter(order, min(SR * 0.45, max(15.0, cutoff)), btype="highpass", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)

def make_dark_reverb_ir(dur_s: float = 3.2) -> np.ndarray:
    """Lush dark Arctic concrete hall stereo impulse response."""
    n_ir = int(dur_s * SR)
    t = np.linspace(0, dur_s, n_ir, dtype=np.float32)
    ir = np.zeros((n_ir, 2), dtype=np.float32)
    for ch in range(2):
        noise = RNG.standard_normal(n_ir).astype(np.float32)
        decay = np.exp(-t * 2.15) * (1.0 - np.exp(-t * 45.0))
        tail = lp_filter(noise * decay, 2600.0, order=2)
        # Early reflections
        for d_ms, g in [(18, 0.35), (37, 0.25), (61, 0.18), (94, 0.12)]:
            idx = int((d_ms + ch * 5) * SR / 1000)
            if idx < n_ir:
                tail[idx] += g * (1 if ch == 0 else -1)
        ir[:, ch] = tail
    ir /= (np.sqrt(np.sum(ir ** 2)) + 1e-9) * 0.85
    return ir

def apply_stereo_reverb(sig_st: np.ndarray, ir: np.ndarray, wet: float = 0.38) -> np.ndarray:
    out = sig_st.copy()
    for ch in range(2):
        conv = signal.fftconvolve(sig_st[:, ch], ir[:, ch], mode="full")[:len(sig_st)]
        out[:, ch] = sig_st[:, ch] * (1.0 - wet * 0.35) + conv * wet
    return out

def add_mono(bus: np.ndarray, t0: float, x: np.ndarray, gain: float = 1.0, pan: float = 0.0):
    """pan in [-1, 1]"""
    i0 = int(round(t0 * SR))
    if i0 >= N:
        return
    if i0 < 0:
        x = x[-i0:]
        i0 = 0
    ln = min(len(x), N - i0)
    if ln <= 0:
        return
    gl = gain * np.cos((pan + 1.0) * np.pi / 4.0)
    gr = gain * np.sin((pan + 1.0) * np.pi / 4.0)
    bus[i0:i0 + ln, 0] += x[:ln] * gl
    bus[i0:i0 + ln, 1] += x[:ln] * gr

def add_stereo(bus: np.ndarray, t0: float, x: np.ndarray, gain: float = 1.0):
    i0 = int(round(t0 * SR))
    if i0 >= N:
        return
    ln = min(len(x), N - i0)
    if ln > 0:
        bus[i0:i0 + ln] += x[:ln] * gain

# ---------- MUSICAL INSTRUMENTS ----------

def synth_cello_drone(dur_s: float, f_start: float, f_end: float = None, vibrato: float = 0.004,
                      brightness: float = 900.0, att: float = 1.2, rel: float = 1.5) -> np.ndarray:
    """Rich bowed dark cello / sub-bass stereo drone with optional smooth glissando."""
    n = int(dur_s * SR)
    t = np.linspace(0, dur_s, n, dtype=np.float64)
    if f_end is None:
        f_end = f_start
    freq = np.linspace(f_start, f_end, n)
    vib = 1.0 + vibrato * np.sin(2 * np.pi * 4.7 * t) * np.minimum(1.0, t / 1.5)
    out = np.zeros((n, 2), dtype=np.float32)
    for ch, detune in enumerate([0.9982, 1.0018]):
        ph = 2 * np.pi * np.cumsum(freq * vib * detune) / SR
        y = np.zeros(n, dtype=np.float64)
        # Bowed string harmonic spectrum + cello body warmth
        for h, amp in [(1, 1.0), (2, 0.62), (3, 0.42), (4, 0.28), (5, 0.16), (6, 0.10), (7, 0.06)]:
            y += amp * np.sin(h * ph + ch * 0.4)
        # Add subtle bow rosin breath
        rosin = bp_filter(RNG.standard_normal(n).astype(np.float32), 600, 2200, 2) * 0.035
        y = lp_filter(y.astype(np.float32) + rosin, brightness, order=2)
        out[:, ch] = y * env_asr(n, att, rel)
    out /= (np.max(np.abs(out)) + 1e-9)
    return out

def synth_piano_note(freq: float, dur_s: float = 5.0) -> np.ndarray:
    """Sparse dark cinematic struck piano note."""
    n = int(dur_s * SR)
    t = np.linspace(0, dur_s, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    B = 0.00018  # piano string stiffness inharmonicity
    for k in range(1, 10):
        fk = k * freq * np.sqrt(1.0 + B * k * k)
        if fk > SR * 0.4:
            break
        decay = np.exp(-t * (0.55 + 0.42 * k))
        amp = (1.0 / (k ** 1.35))
        y += amp * decay * np.sin(2 * np.pi * fk * t)
    # Soft felt hammer thud
    thud = lp_filter(RNG.standard_normal(min(n, int(0.04 * SR))).astype(np.float32), 320, 2)
    y[:len(thud)] += thud * np.exp(-np.linspace(0, 12, len(thud))) * 0.18
    # 5ms attack
    att = int(0.005 * SR)
    y[:att] *= np.linspace(0, 1, att)
    y = y.astype(np.float32)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def synth_pulse_bar(dur_s: float, bpm: float, root_hz: float, pattern_mults, cutoff_start=360.0,
                    cutoff_end=680.0, decay_s=0.22) -> np.ndarray:
    """Minimalist dark analog synth pulse (LEMMiNO / Vibe-SCI style)."""
    n = int(dur_s * SR)
    out = np.zeros((n, 2), dtype=np.float32)
    step_s = 60.0 / bpm / 2.0  # eighth-note pulse
    t_cur = 0.0
    step_idx = 0
    while t_cur < dur_s - 0.05:
        mult = pattern_mults[step_idx % len(pattern_mults)]
        if mult > 0:
            f = root_hz * mult
            ln = int(min(decay_s, dur_s - t_cur) * SR)
            t = np.linspace(0, ln / SR, ln, dtype=np.float64)
            # Warm analog saw+sub
            ph = 2 * np.pi * f * t
            raw = 0.65 * np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.22 * np.sin(3 * ph) + 0.12 * np.sin(4 * ph)
            prog = t_cur / max(1e-6, dur_s)
            cut = cutoff_start + (cutoff_end - cutoff_start) * prog
            accent = 1.0 if (step_idx % 4 == 0) else 0.62
            note = lp_filter(raw.astype(np.float32), cut, 2) * env_ad(ln, 0.006, ln / SR - 0.006) * accent
            pan = -0.18 if (step_idx % 2 == 0) else 0.18
            i0 = int(t_cur * SR)
            ln2 = min(ln, n - i0)
            out[i0:i0 + ln2, 0] += note[:ln2] * (0.75 - pan * 0.25)
            out[i0:i0 + ln2, 1] += note[:ln2] * (0.75 + pan * 0.25)
        t_cur += step_s
        step_idx += 1
    out /= (np.max(np.abs(out)) + 1e-9)
    return out

def synth_sub_impact(dur_s: float = 4.0, f_top: float = 55.0, f_bot: float = 28.0) -> np.ndarray:
    """Deep cinema sub-bass hit + hollow metallic borehole pipe drop."""
    n = int(dur_s * SR)
    t = np.linspace(0, dur_s, n, dtype=np.float64)
    freq = f_bot + (f_top - f_bot) * np.exp(-t * 4.5)
    ph = 2 * np.pi * np.cumsum(freq) / SR
    sub = np.sin(ph) * np.exp(-t * 1.15)
    # Initial sharp transient thump
    punch = lp_filter(RNG.standard_normal(n).astype(np.float32), 240, 2) * np.exp(-t * 28.0) * 0.9
    # Hollow steel borehole metallic modes
    metal = np.zeros(n, dtype=np.float64)
    for fm, dec, g in [(186.0, 2.2, 0.28), (317.0, 2.8, 0.20), (493.0, 3.6, 0.14), (782.0, 5.0, 0.09)]:
        metal += g * np.sin(2 * np.pi * fm * t) * np.exp(-t * dec)
    y = (sub + punch + metal * 0.75).astype(np.float32)
    y[:int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def synth_metal_clang(dur_s: float = 1.6, base_hz: float = 240.0) -> np.ndarray:
    n = int(dur_s * SR)
    t = np.linspace(0, dur_s, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for mult, dec, g in [(1.0, 3.0, 0.5), (1.58, 3.8, 0.4), (2.41, 4.8, 0.3), (3.73, 6.2, 0.2), (5.12, 8.5, 0.12)]:
        y += g * np.sin(2 * np.pi * base_hz * mult * t) * np.exp(-t * dec)
    hit = bp_filter(RNG.standard_normal(n).astype(np.float32), 400, 3500, 2) * np.exp(-t * 35.0) * 0.5
    y = (y + hit).astype(np.float32)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def synth_wind(dur_s: float, low_hz: float = 65.0, high_hz: float = 520.0, gust_hz: float = 0.22) -> np.ndarray:
    n = int(dur_s * SR)
    t = np.linspace(0, dur_s, n, dtype=np.float32)
    out = np.zeros((n, 2), dtype=np.float32)
    for ch in range(2):
        p = pink_noise(n)
        w1 = bp_filter(p, low_hz, high_hz, 2)
        gust = 0.55 + 0.45 * np.sin(2 * np.pi * gust_hz * t + ch * 1.3 + 0.4 * np.sin(2 * np.pi * 0.09 * t))
        # Subtle howling resonance
        howl = bp_filter(p, low_hz * 2.8, low_hz * 4.2, 3) * (0.3 + 0.7 * np.maximum(0, np.sin(2 * np.pi * (gust_hz * 0.7) * t + ch)))
        out[:, ch] = (w1 * gust + howl * 0.35) * env_asr(n, 0.6, 0.8)
    out /= (np.max(np.abs(out)) + 1e-9)
    return out

def synth_hell_tape(dur_s: float = 16.0) -> np.ndarray:
    """Eerie lo-fi tape-wobbled formant-filtered haunting sound (NOT cheesy horror, authentic degraded tape)."""
    n = int(dur_s * SR)
    t = np.linspace(0, dur_s, n, dtype=np.float64)
    # Slow tape wow & flutter
    wow = 1.0 + 0.025 * np.sin(2 * np.pi * 0.65 * t) + 0.008 * np.sin(2 * np.pi * 5.2 * t)
    # Multi-voice eerie distant wail cluster (minor 2nds / tritones)
    src = np.zeros(n, dtype=np.float64)
    for f0, ph0 in [(196.0, 0.0), (208.0, 1.1), (277.0, 2.3), (311.0, 0.7)]:
        f_mod = f0 * wow * (1.0 + 0.09 * np.sin(2 * np.pi * 0.19 * t + ph0))
        ph = 2 * np.pi * np.cumsum(f_mod) / SR
        # Rich harmonic buzz
        v = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.35 * np.sin(3 * ph) + 0.2 * np.sin(4 * ph)
        swell = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.27 * t + ph0 * 2.1))
        src += v * swell
    # Add turbulent breathy noise
    src = src.astype(np.float32) * 0.35 + bp_filter(pink_noise(n), 280, 1800, 2) * 0.45
    # Vocal-tract-like bandpass formants + heavy lo-fi bandwidth limit (300-2200 Hz)
    f1 = bp_filter(src, 340, 720, 3)
    f2 = bp_filter(src, 950, 1650, 3)
    y = f1 * 0.75 + f2 * 0.45
    # Analog cassette saturation (soft clip, then harder overload around 2.0-3.0s for Shot 01 clip visual!)
    overload = np.ones(n, dtype=np.float32)
    i_ov0, i_ov1 = int(1.9 * SR), int(3.0 * SR)
    overload[i_ov0:i_ov1] = np.linspace(1.0, 3.2, i_ov1 - i_ov0)
    y = np.tanh(y * 1.4 * overload) * 0.85
    # Add cassette hiss
    hiss = bp_filter(RNG.standard_normal(n).astype(np.float32), 2500, 7500, 2) * 0.055
    y = lp_filter(y + hiss, 3800, 2)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def synth_click(dur_ms: float = 18.0, hp_hz: float = 1400.0) -> np.ndarray:
    n = int(dur_ms * SR / 1000)
    t = np.linspace(0, dur_ms / 1000, n, dtype=np.float32)
    x = hp_filter(RNG.standard_normal(n).astype(np.float32), hp_hz, 2) * np.exp(-t * 240.0)
    return x / (np.max(np.abs(x)) + 1e-9)

def synth_water_drop(f0: float = 1650.0) -> np.ndarray:
    dur = 0.14
    n = int(dur * SR)
    t = np.linspace(0, dur, n, dtype=np.float64)
    # Classic rising Minnaert bubble pitch!
    f = f0 * (1.0 + 0.55 * (1.0 - np.exp(-t * 40.0)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = (np.sin(ph) * np.exp(-t * 45.0)).astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9)

def synth_mud_bubble(f0: float = 135.0) -> np.ndarray:
    dur = 0.22
    n = int(dur * SR)
    t = np.linspace(0, dur, n, dtype=np.float64)
    f = f0 * (1.0 + 0.9 * (1.0 - np.exp(-t * 25.0)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = (np.sin(ph) * np.exp(-t * 18.0)).astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9)

def synth_relay_switch() -> np.ndarray:
    """Crisp Soviet heavy electromagnetic relay clack."""
    dur = 0.08
    n = int(dur * SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    imp = bp_filter(RNG.standard_normal(n).astype(np.float32), 600, 4200, 2) * np.exp(-t * 160.0)
    body = np.sin(2 * np.pi * 420 * t) * np.exp(-t * 90.0) * 0.45
    y = imp + body
    return y / (np.max(np.abs(y)) + 1e-9)

def synth_pipe_fracture_climax() -> np.ndarray:
    """Shot 19 climax: metal torsion groan -> violent snap -> 5km steel pipe falling down abyss + sub drop."""
    dur = 8.5
    n = int(dur * SR)
    t = np.linspace(0, dur, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float32)

    # 0.0 - 1.6s: agonizing metal torsion groan
    n1 = int(1.6 * SR)
    t1 = t[:n1]
    groan = np.sin(2 * np.pi * np.cumsum(np.linspace(115, 290, n1)) / SR)
    groan += 0.5 * np.sin(2 * np.pi * np.cumsum(np.linspace(268, 610, n1)) / SR)
    groan *= (t1 / 1.6) ** 2 * 0.45
    y[:n1] += groan.astype(np.float32)

    # 1.6s: VIOLENT SNAP + SUB DROP (60 -> 24 Hz)
    snap_i = int(1.6 * SR)
    rem = n - snap_i
    tr = np.linspace(0, rem / SR, rem, dtype=np.float64)
    crack = bp_filter(RNG.standard_normal(rem).astype(np.float32), 500, 6000, 2) * np.exp(-tr * 26.0) * 1.4
    sub_f = 24.0 + 42.0 * np.exp(-tr * 1.6)
    sub = np.sin(2 * np.pi * np.cumsum(sub_f) / SR) * np.exp(-tr * 0.55) * 1.1

    # Falling hollow steel pipe whistling down the 12km dark borehole (pitch dropping continuously)
    pipe_f = 520.0 * np.exp(-tr * 0.34) + 75.0
    pipe_ph = 2 * np.pi * np.cumsum(pipe_f) / SR
    pipe = (0.45 * np.sin(pipe_ph) + 0.30 * np.sin(1.62 * pipe_ph) + 0.20 * np.sin(2.48 * pipe_ph))
    # Intermittent wall-scraping rattles as the 5km steel column drops into blackness
    rattle_env = np.exp(-tr * 0.42) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * tr) ** 2)
    pipe_sig = (pipe * rattle_env * 0.48).astype(np.float32)

    y[snap_i:] += (crack + sub + pipe_sig)
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def main():
    print("Synthesizing 300s 3-layer cinema soundscape (Ambience + Foley + Score)...")
    amb = np.zeros((N, 2), dtype=np.float32)
    fol = np.zeros((N, 2), dtype=np.float32)
    sco = np.zeros((N, 2), dtype=np.float32)
    reverb_send = np.zeros((N, 2), dtype=np.float32)

    # Load narration timeline for exact cue alignment!
    tl = json.load(open(f"{AUDIO_DIR}/narration_timeline.json", encoding="utf-8"))
    shots_by_id = {s["shot"]: s for s in tl["shots"]}

    # ==========================================================
    # ACT 1: 传说的坍塌 (00:00 - 00:40)
    # ==========================================================
    # Shot 01-03 tape hiss + eerie hell recording
    hell_tape = synth_hell_tape(16.0)
    # Save exact envelope & audio for Shot 01 live waveform & Shot 03 live spectrogram!
    np.save(f"{AUDIO_DIR}/hell_tape_mono.npy", hell_tape.astype(np.float32))

    # Shot 01 (0-3s): Tape hiss + eerie wail + 25Hz sub swell
    add_mono(fol, 0.0, hell_tape[:int(15.0 * SR)], gain=0.22, pan=0.0)
    # Boost Shot 01 (0-3s) slightly when no narrator, duck in Shot 02 (3-15s) as TV audio
    fol[:int(3.0 * SR)] *= 1.85
    # Sub 25 Hz rising in Shot 01
    t1 = np.linspace(0, 3.0, int(3.0 * SR), dtype=np.float32)
    sub1 = np.sin(2 * np.pi * 28.0 * t1) * np.linspace(0, 1, len(t1)) ** 1.8
    add_mono(sco, 0.0, sub1, gain=0.28)

    # Shot 02 (3-15s): CRT whine (subtle 11kHz muted) + room hum + C1 drone + dissonant Db1 tension
    t2 = np.linspace(0, 12.0, int(12.0 * SR), dtype=np.float32)
    crt = np.sin(2 * np.pi * 9800.0 * t2) * 0.008 + lp_filter(pink_noise(len(t2)), 180, 2) * 0.06
    add_mono(amb, 3.0, crt * env_asr(len(t2), 0.2, 0.2), gain=0.45)
    add_mono(fol, 3.2, synth_click(25, 900), gain=0.22, pan=-0.2)
    add_stereo(sco, 3.0, synth_cello_drone(12.5, 32.70, 32.70, vibrato=0.002, brightness=380, att=1.0, rel=0.4), gain=0.30)
    add_stereo(sco, 5.0, synth_cello_drone(10.0, 69.30, 69.30, vibrato=0.004, brightness=550, att=2.0, rel=0.5), gain=0.14)  # Db2 minor-2nd unease

    # Shot 03 (15-28s): Reversed & slowed tape excerpt, then crisp digital cut around "取自" (~20.5s)!
    # Find exact timestamp of "那段录音" in Shot 03
    s3_events = [e for e in shots_by_id[3]["events"] if e["type"] == "speech"]
    cut_t = s3_events[-1]["start"] + 1.4 if len(s3_events) >= 2 else 21.0
    # Digital static bed
    add_mono(amb, 15.0, bp_filter(RNG.standard_normal(int(13 * SR)).astype(np.float32), 1800, 7000, 2) * 0.025, gain=0.5)
    # Reversed & stretched tape fragment from 15.0 to cut_t
    frag_dur = max(2.0, cut_t - 15.0)
    frag_raw = hell_tape[int(1.0 * SR):int(5.5 * SR)][::-1]
    frag_slow = signal.resample(frag_raw, int(frag_dur * SR)).astype(np.float32)
    add_mono(fol, 15.0, frag_slow * env_asr(len(frag_slow), 0.1, 0.05), gain=0.22)
    # Sharp digital cut click right at cut_t!
    add_mono(fol, cut_t, synth_click(12, 2200), gain=0.45)
    # Drone drops sharply right at cut_t (leaving hollow room)
    add_stereo(sco, 15.0, synth_cello_drone(max(1.0, cut_t - 15.0), 32.70, 32.70, brightness=320, att=0.1, rel=0.08), gain=0.24)

    # Shot 04 (28-36s): 40 Hz Arctic wind howling + clanging rusted roof sheet + C2 cello entering 0.5s before "井，是真的"
    add_stereo(amb, 28.0, synth_wind(8.2, low_hz=42.0, high_hz=480.0, gust_hz=0.32), gain=0.38)
    for tc in [28.7, 30.8, 32.9, 34.8]:
        add_mono(fol, tc, synth_metal_clang(1.4, base_hz=175.0 + RNG.uniform(-15, 15)), gain=0.16, pan=0.35)
    s4_speech = [e for e in shots_by_id[4]["events"] if e["type"] == "speech"]
    well_t = (s4_speech[-1]["start"] - 0.5) if s4_speech else 33.0
    cello4 = synth_cello_drone(max(1.5, 36.2 - well_t), 65.41, 65.41, vibrato=0.005, brightness=950, att=0.45, rel=0.4)
    add_stereo(reverb_send, well_t, cello4, gain=0.36)

    # Shot 05 (36-40s): TITLE HIT @ 00:36.00! Massive 30Hz sub + steel pipe into borehole, then decay to silence
    add_stereo(amb, 36.0, synth_wind(3.5, low_hz=45, high_hz=260), gain=0.06)
    impact5 = synth_sub_impact(3.8, f_top=62.0, f_bot=29.0)
    add_mono(fol, 36.00, impact5, gain=0.88)
    add_mono(reverb_send, 36.00, impact5, gain=0.45)
    # 00:37.00 faint steel cable lowering friction
    t_cab = np.linspace(0, 2.4, int(2.4 * SR), dtype=np.float32)
    cable = bp_filter(pink_noise(len(t_cab)), 900, 2800, 3) * 0.045 * env_asr(len(t_cab), 0.2, 0.5)
    add_mono(fol, 37.0, cable, gain=0.55)

    # ==========================================================
    # ACT 2: 垂直的野心 (00:40 - 01:35)
    # ==========================================================
    # Shot 06 (40-55s): Open Arctic wind + distant 50Hz Soviet diesel hum + rigging clanks;
    # Moon insert (~49.2-51.5s) VACUUMS OUT all sound to Apollo radio hiss!
    s6_speech = [e for e in shots_by_id[6]["events"] if e["type"] == "speech"]
    moon_t0 = s6_speech[-1]["start"] - 0.15 if len(s6_speech) >= 2 else 49.8
    moon_t1 = min(54.2, moon_t0 + 2.3)

    wind6 = synth_wind(15.0, low_hz=55, high_hz=380, gust_hz=0.18)
    # Distant diesel generator 50Hz + 100Hz hum
    td6 = np.linspace(0, 15.0, int(15.0 * SR), dtype=np.float32)
    diesel = (np.sin(2 * np.pi * 50 * td6) * 0.5 + np.sin(2 * np.pi * 100 * td6) * 0.35 + np.sin(2 * np.pi * 150 * td6) * 0.15) * 0.09
    wind6[:, 0] += diesel
    wind6[:, 1] += diesel
    # Carve total vacuum during lunar insert!
    m0, m1 = int((moon_t0 - 40.0) * SR), int((moon_t1 - 40.0) * SR)
    fade_v = int(0.04 * SR)
    if 0 < m0 < m1 < len(wind6):
        wind6[m0:m1] *= 0.0
        wind6[max(0, m0 - fade_v):m0] *= np.linspace(1, 0, min(fade_v, m0))[:, None]
        wind6[m1:min(len(wind6), m1 + fade_v)] *= np.linspace(0, 1, min(fade_v, len(wind6) - m1))[:, None]
    add_stereo(amb, 40.0, wind6, gain=0.24)

    # Lunar radio static during moon_t0..moon_t1
    n_moon = int((moon_t1 - moon_t0) * SR)
    radio = bp_filter(RNG.standard_normal(n_moon).astype(np.float32), 1100, 3200, 3) * 0.05
    add_mono(amb, moon_t0, radio * env_asr(n_moon, 0.03, 0.03), gain=0.7)
    # Metallic rigging clanks before lunar cut
    for tc in [41.2, 44.5, 47.1]:
        add_mono(fol, tc, synth_metal_clang(1.0, 320.0), gain=0.10, pan=-0.3)

    # 60 BPM analog pulse enters softly at 00:44.0 (muted during lunar cut!)
    pulse6 = synth_pulse_bar(11.0, bpm=60.0, root_hz=65.41, pattern_mults=[1, 0, 1, 1.5, 1, 0, 1.189, 1], cutoff_start=260, cutoff_end=420)
    pm0, pm1 = int((moon_t0 - 44.0) * SR), int((moon_t1 - 44.0) * SR)
    if 0 < pm0 < pm1 < len(pulse6):
        pulse6[pm0:pm1] = 0.0
    add_stereo(sco, 44.0, pulse6, gain=0.18)

    # Shot 07 (55-66s): Electronic hum + odometer ticks + pulse dropping an octave on "向下"
    add_stereo(amb, 55.0, synth_cello_drone(11.0, 65.41, 65.41, vibrato=0.0, brightness=240, att=0.4, rel=0.4), gain=0.12)
    s7_speech = [e for e in shots_by_id[7]["events"] if e["type"] == "speech"]
    down_t = s7_speech[2]["start"] if len(s7_speech) >= 3 else 60.5
    add_stereo(sco, 55.0, synth_pulse_bar(max(1.0, down_t - 55.0), 60.0, 65.41, [1, 1, 1.5, 1], 360, 480), gain=0.22)
    add_stereo(sco, down_t, synth_pulse_bar(max(1.0, 66.0 - down_t), 60.0, 32.70, [1, 0, 1, 1.5], 320, 460), gain=0.28)
    # Odometer rapid mechanical clicks during 384,400 km (55.5-57.5) and -15,000 m (down_t..down_t+2.5)
    for tc in np.linspace(55.4, 57.6, 18):
        add_mono(fol, tc, synth_click(8, 2600), gain=0.09)
    for tc in np.linspace(down_t + 0.1, min(64.5, down_t + 2.4), 22):
        add_mono(fol, tc, synth_click(8, 2200), gain=0.11)

    # Shot 08 (66-82s): Deep seismic rumble + pulse + double bass pizzicato + Mohole wave/projector + RED STAMP THUD
    t8 = np.linspace(0, 10.0, int(10.0 * SR), dtype=np.float32)
    seismic = lp_filter(pink_noise(len(t8)), 85, 2) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t8))
    add_mono(amb, 66.0, seismic * env_asr(len(t8), 0.5, 0.6), gain=0.26)
    # Archival ocean waves + film projector chatter (76-82s)
    waves = lp_filter(pink_noise(int(6.0 * SR)), 420, 2) * env_asr(int(6.0 * SR), 0.3, 0.3)
    add_mono(amb, 76.0, waves, gain=0.18)
    for tc in np.arange(76.0, 81.8, 1.0 / 18.0):
        add_mono(fol, tc, synth_click(5, 3200), gain=0.028)
    # Red stamp "PROJECT MOHOLE · 1966 · CANCELLED" thud @ 79.4s
    add_mono(fol, 79.4, synth_sub_impact(1.2, 95, 38), gain=0.55)
    add_stereo(sco, 66.0, synth_pulse_bar(16.0, 62.0, 65.41, [1, 0, 1.189, 1, 1.5, 0, 1.189, 1], 400, 560), gain=0.24)

    # Shot 09 (82-95s): Heavy rig diesel roar + hydraulic hiss + steel clanks + mud pump chug; pulse accelerates to 72 BPM -> ABRUPT STOP at 93.0s!
    t9 = np.linspace(0, 11.2, int(11.2 * SR), dtype=np.float32)
    rig_roar = lp_filter(pink_noise(len(t9)), 240, 2) * 0.55 + np.sin(2 * np.pi * 50 * t9) * 0.25 + np.sin(2 * np.pi * 100 * t9) * 0.15
    # Rhythmic mud-pump chug (1.6 Hz)
    pump = 0.72 + 0.28 * np.sin(2 * np.pi * 1.6 * t9) ** 2
    add_mono(amb, 82.0, rig_roar * pump * env_asr(len(t9), 0.2, 0.35), gain=0.32)
    # Hydraulic hisses & steel clanks
    for tc in [82.6, 85.4, 88.2, 90.8]:
        add_mono(fol, tc, synth_metal_clang(1.2, 290.0), gain=0.22, pan=RNG.uniform(-0.3, 0.3))
        hiss = bp_filter(RNG.standard_normal(int(0.65 * SR)).astype(np.float32), 1800, 6500, 2) * env_ad(int(0.65 * SR), 0.04, 0.60)
        add_mono(fol, tc + 0.25, hiss, gain=0.14, pan=-0.2)
    add_stereo(sco, 82.0, synth_pulse_bar(11.0, 72.0, 65.41, [1, 1, 1.5, 1, 1.189, 1, 1.5, 1.335], 500, 780), gain=0.28)
    # 93.0 - 95.0: sudden vacuum drop into Act 3 silence!

    # ==========================================================
    # ACT 3: 深度的反噬 (01:35 - 02:35)
    # ==========================================================
    # Shot 10 (95-106s): Near silence + single sparse low piano note C2 (65.41 Hz)
    add_mono(reverb_send, 95.8, synth_piano_note(65.41, 6.5), gain=0.52)
    add_mono(reverb_send, 102.2, synth_piano_note(98.00, 5.0), gain=0.30)  # G2 subtle fifth

    # Shot 11 (106-126s): Thermal low hum + RISING SINE SWEEP tracking 180 °C curve + cello glissando C2 -> D2!
    s11_sp = [e for e in shots_by_id[11]["events"] if e["type"] == "speech"]
    t_180_start = s11_sp[1]["start"] if len(s11_sp) >= 2 else 112.0
    t_180_lock = s11_sp[1]["end"] if len(s11_sp) >= 2 else 115.0
    # Thermal hum throughout Shot 11
    add_stereo(amb, 106.0, synth_cello_drone(20.0, 48.0, 54.0, vibrato=0.0, brightness=160, att=1.0, rel=1.0), gain=0.18)
    # Rising pure eerie sine sweep synced with real curve (110.5s -> t_180_lock), then holds tense unresolved pitch!
    n_sw = int((125.8 - 110.5) * SR)
    t_sw = np.linspace(0, 125.8 - 110.5, n_sw, dtype=np.float64)
    ramp_dur = max(1.5, t_180_lock - 110.5)
    f_sw = np.where(t_sw < ramp_dur, 220.0 + (466.16 - 220.0) * (t_sw / ramp_dur) ** 1.35, 466.16)
    sw_sig = (np.sin(2 * np.pi * np.cumsum(f_sw) / SR) * env_asr(n_sw, 0.5, 1.2)).astype(np.float32)
    add_mono(reverb_send, 110.5, sw_sig, gain=0.11)
    # Cello + sub drone glissando UP one whole tone (65.41 Hz C2 -> 73.42 Hz D2)!
    cello11 = synth_cello_drone(19.5, 65.41, 73.42, vibrato=0.005, brightness=980, att=1.5, rel=1.5)
    add_stereo(reverb_send, 106.5, cello11, gain=0.34)
    # Deep warning pulse when 180°C locks
    add_mono(fol, t_180_lock, synth_sub_impact(2.5, 58, 32), gain=0.45)

    # Shot 12 (126-141s): Lithostatic deep rumble + pitch-dropped rock/ice creak + cello vibrato sustain
    t12 = np.linspace(0, 15.0, int(15.0 * SR), dtype=np.float32)
    rumble12 = lp_filter(pink_noise(len(t12)), 95, 2) * env_asr(len(t12), 0.6, 0.8)
    add_mono(amb, 126.0, rumble12, gain=0.34)
    # Slow viscous rock/ice pressure groans
    for tg in [127.5, 131.8, 136.0]:
        ng = int(3.2 * SR)
        tt = np.linspace(0, 3.2, ng, dtype=np.float64)
        gr = bp_filter(pink_noise(ng), 140, 340, 4) * np.sin(2 * np.pi * 62 * tt) * env_asr(ng, 0.7, 1.0)
        add_mono(fol, tg, gr, gain=0.30, pan=RNG.uniform(-0.35, 0.35))
    add_stereo(reverb_send, 126.0, synth_cello_drone(15.2, 73.42, 73.42, vibrato=0.008, brightness=880, att=0.8, rel=1.2), gain=0.32)

    # Shot 13 (141-155s): Claustrophobic closing-filter pressure drone + metal scraping + wet mud squeeze
    n13 = int(14.0 * SR)
    p13 = pink_noise(n13)
    # Time-varying closing lowpass via crossfade of 3 bands (evokes walls creeping inward!)
    w_hi = lp_filter(p13, 620, 2)
    w_lo = lp_filter(p13, 110, 2)
    prog13 = np.linspace(0, 1, n13, dtype=np.float32)
    squeeze = (w_hi * (1.0 - prog13) + w_lo * (0.4 + 0.6 * prog13)) * env_asr(n13, 0.4, 1.2)
    add_mono(amb, 141.0, squeeze, gain=0.34)
    # Scraping rock/steel when bit re-descends (149-151.5s)
    n_scr = int(2.6 * SR)
    scrape = bp_filter(pink_noise(n_scr), 380, 1400, 3) * env_asr(n_scr, 0.2, 0.5)
    add_mono(fol, 148.8, scrape, gain=0.16)
    add_stereo(sco, 141.0, synth_cello_drone(14.0, 73.42, 65.41, vibrato=0.003, brightness=520, att=0.5, rel=2.0), gain=0.22)

    # ==========================================================
    # ACT 4: 被改写的教科书 (02:35 - 03:35)
    # ==========================================================
    # Shot 14 (155-162s): Core archive room tone + wooden core thump + 72 BPM metronomic pulse
    add_mono(amb, 155.0, lp_filter(pink_noise(int(7.0 * SR)), 180, 2) * 0.08 * env_asr(int(7 * SR), 0.3, 0.3), gain=0.6)
    add_mono(fol, 156.2, synth_sub_impact(0.45, 160, 72), gain=0.26)  # wooden core box thump
    # Clinical 72 BPM pulse in Db major color (Db2 = 69.30 Hz) through Shots 14-17 (155.5 - 196.8s)
    # Drops 2 beats right after "是氢气" in Shot 17!
    pulse14_17 = synth_pulse_bar(42.0, bpm=72.0, root_hz=69.30,
                                 pattern_mults=[1, 0, 1.5, 1, 1.26, 0, 1.5, 1],
                                 cutoff_start=340, cutoff_end=640, decay_s=0.18)
    # Mute 2 beats (~1.66s) around 196.2s after "是氢气"
    s17_sp = [e for e in shots_by_id[17]["events"] if e["type"] == "speech"]
    h2_end = s17_sp[-1]["end"] if s17_sp else 196.2
    hp0 = int((h2_end - 155.5) * SR)
    if 0 < hp0 < len(pulse14_17):
        pulse14_17[hp0:] = 0.0
    add_stereo(sco, 155.5, pulse14_17, gain=0.20)

    # Shot 15 (162-176s): Seismic throb + pencil scratch strikeout on "没有玄武岩" + piano single notes every 4 beats
    add_mono(amb, 162.0, lp_filter(pink_noise(int(14.0 * SR)), 90, 2) * env_asr(int(14 * SR), 0.3, 0.4), gain=0.22)
    s15_sp = [e for e in shots_by_id[15]["events"] if e["type"] == "speech"]
    nobasalt_t = s15_sp[3]["start"] if len(s15_sp) >= 4 else 169.5
    # Crisp stylus/pencil scratch across paper
    n_pen = int(0.32 * SR)
    pen = bp_filter(RNG.standard_normal(n_pen).astype(np.float32), 2200, 6800, 2) * env_ad(n_pen, 0.02, 0.29)
    add_mono(fol, nobasalt_t, pen, gain=0.28, pan=0.15)
    # Piano single notes in Db major color (Db3, F3, Ab3, Db3)
    for tp, fp in [(162.5, 138.59), (165.83, 174.61), (169.16, 207.65), (172.5, 138.59)]:
        add_mono(reverb_send, tp, synth_piano_note(fp, 3.8), gain=0.30, pan=-0.1)

    # Shot 16 (176-189s): Deep rock pressure + delicate crystalline water droplet pings + high-octave piano
    add_mono(amb, 176.0, lp_filter(pink_noise(int(13.0 * SR)), 110, 2) * env_asr(int(13 * SR), 0.3, 0.4), gain=0.20)
    for td, fd in [(178.4, 1520), (180.6, 1890), (182.9, 1410), (184.5, 2050), (186.1, 1680), (187.8, 1920)]:
        add_mono(reverb_send, td, synth_water_drop(fd), gain=0.16, pan=RNG.uniform(-0.4, 0.4))
    for tp, fp in [(176.5, 277.18), (179.83, 349.23), (183.16, 415.30), (186.5, 277.18)]:
        add_mono(reverb_send, tp, synth_piano_note(fp, 4.0), gain=0.26, pan=0.12)

    # Shot 17 (189-199s): Distant rig hum + thick viscous bubbling mud pops
    add_mono(amb, 189.0, lp_filter(pink_noise(int(10.0 * SR)), 210, 2) * env_asr(int(10 * SR), 0.3, 0.5), gain=0.22)
    for tb in np.arange(189.3, 198.4, 0.38):
        add_mono(fol, tb + RNG.uniform(-0.1, 0.1), synth_mud_bubble(RNG.uniform(95, 175)), gain=0.22, pan=RNG.uniform(-0.35, 0.35))

    # Shot 18 (199-215s): Microscope lamp faint hum + focus dial micro-click + lone shimmering high piano note
    t18 = np.linspace(0, 16.0, int(16.0 * SR), dtype=np.float32)
    lamp = (np.sin(2 * np.pi * 100 * t18) * 0.4 + np.sin(2 * np.pi * 200 * t18) * 0.25) * 0.03 * env_asr(len(t18), 0.4, 0.6)
    add_mono(amb, 199.0, lamp, gain=0.7)
    for tc in [200.2, 201.0, 201.9]:
        add_mono(fol, tc, synth_click(10, 3400), gain=0.10)
    # Ethereal high piano single notes in huge reverb (Ab4 = 415.30, F4 = 349.23, Db4 = 277.18)
    add_mono(reverb_send, 199.8, synth_piano_note(415.30, 6.5), gain=0.34, pan=0.05)
    add_mono(reverb_send, 206.5, synth_piano_note(349.23, 6.0), gain=0.28, pan=-0.05)
    add_mono(reverb_send, 211.8, synth_piano_note(277.18, 5.0), gain=0.30, pan=0.0)

    # ==========================================================
    # ACT 5: 断裂、遗弃与余韵 (03:35 - 05:00)
    # ==========================================================
    # Shot 19 (215-227s): DRILL STRING FRACTURE CLIMAX!
    s19_sp = [e for e in shots_by_id[19]["events"] if e["type"] == "speech"]
    snap_cue = (s19_sp[1]["start"] + 0.5) if len(s19_sp) >= 2 else 219.5
    # Rig hum abruptly cuts off right before fracture!
    pre_len = max(0.5, snap_cue - 215.0)
    add_mono(amb, 215.0, lp_filter(pink_noise(int(pre_len * SR)), 220, 2) * env_asr(int(pre_len * SR), 0.1, 0.08), gain=0.28)
    # Trigger torsion -> snap -> falling pipe -> sub drop!
    climax = synth_pipe_fracture_climax()
    add_mono(fol, snap_cue - 1.6, climax, gain=0.82)
    add_mono(reverb_send, snap_cue - 1.6, climax, gain=0.40)

    # Shot 20 (227-240s): Depth counter clicks -> lock chime + low impact on 12,262 m + cello/drone return
    add_stereo(amb, 227.0, synth_cello_drone(13.0, 65.41, 65.41, vibrato=0.0, brightness=280, att=0.8, rel=0.6), gain=0.14)
    s20_sp = [e for e in shots_by_id[20]["events"] if e["type"] == "speech"]
    lock_t = (s20_sp[2]["end"] - 0.5) if len(s20_sp) >= 3 else 235.5
    for tc in np.linspace(231.0, lock_t - 0.1, 24):
        add_mono(fol, tc, synth_click(8, 2500), gain=0.10)
    # Clean lock ping + low impact on 12,262 m
    add_mono(reverb_send, lock_t, synth_piano_note(523.25, 3.0), gain=0.22)  # C5 clear lock tone
    add_mono(fol, lock_t, synth_sub_impact(2.5, 58, 32.7), gain=0.55)
    add_stereo(reverb_send, 229.0, synth_cello_drone(11.5, 65.41, 65.41, vibrato=0.004, brightness=780, att=1.5, rel=0.8), gain=0.28)

    # Shot 21 (240-252s): Control room hum stepping down + 8 crisp Soviet relay clicks (245.5 -> 250.4s) + descending cello
    t21 = np.linspace(0, 12.0, int(12.0 * SR), dtype=np.float32)
    ctrl_hum = (np.sin(2 * np.pi * 50 * t21) * 0.5 + np.sin(2 * np.pi * 150 * t21) * 0.25) * np.linspace(1.0, 0.05, len(t21))
    add_mono(amb, 240.0, ctrl_hum * 0.14, gain=0.8)
    # Save exact 8 relay timestamps so Shot 21b visual lamps switch off IN PERFECT SYNC!
    relay_times = [round(245.4 + i * 0.65, 3) for i in range(8)]
    for i, tr in enumerate(relay_times):
        add_mono(fol, tr, synth_relay_switch(), gain=0.34, pan=-0.5 + i * (1.0 / 7.0))
    # Cello slowly descending (G2 -> C2) fading into wind
    add_stereo(reverb_send, 240.0, synth_cello_drone(10.5, 98.00, 65.41, vibrato=0.004, brightness=680, att=0.6, rel=2.0), gain=0.26)
    add_stereo(amb, 249.5, synth_wind(3.0, 70, 560), gain=0.18)

    # Shot 22 (252-260s): High wind whistle through shattered concrete windows + loose metal rattle + faint drone
    add_stereo(amb, 252.0, synth_wind(8.2, low_hz=120, high_hz=1350, gust_hz=0.28), gain=0.32)
    add_mono(fol, 254.4, synth_metal_clang(1.3, 210), gain=0.11, pan=0.45)
    add_mono(fol, 258.1, synth_metal_clang(1.1, 185), gain=0.09, pan=-0.35)
    add_stereo(sco, 252.0, synth_cello_drone(8.2, 65.41, 65.41, vibrato=0.002, brightness=320, att=0.5, rel=0.8), gain=0.14)

    # Shot 23 (260-271s): Wind recedes + tiny snow grains tapping iron cap + single sparse C3 piano note
    add_stereo(amb, 260.0, synth_wind(11.0, low_hz=60, high_hz=420, gust_hz=0.15), gain=0.16)
    for ts in [261.4, 263.1, 264.8, 266.2, 267.9, 269.5]:
        add_mono(fol, ts, synth_click(6, 4200), gain=0.06, pan=RNG.uniform(-0.3, 0.3))
    add_mono(reverb_send, 268.2, synth_piano_note(130.81, 6.5), gain=0.40)  # C3 lone note in leave-space

    # Shot 24 (271-284s): Vacuum quiet + slow planetary zoom air whoosh + lush dark string chord blooming (C minor -> open 5th)
    n_wh = int(5.5 * SR)
    whoosh = bp_filter(pink_noise(n_wh), 90, 620, 2) * env_asr(n_wh, 2.2, 2.5)
    add_mono(fol, 272.5, whoosh, gain=0.18)
    # Slow lush dark string chord: C2 (65.41) + G2 (98.00) + Eb3 (155.56) + C3 (130.81)
    add_stereo(reverb_send, 272.0, synth_cello_drone(13.0, 65.41, 65.41, 0.004, 720, att=2.8, rel=1.5), gain=0.28)
    add_stereo(reverb_send, 273.5, synth_cello_drone(11.5, 130.81, 130.81, 0.004, 920, att=2.8, rel=1.5), gain=0.22)
    add_stereo(reverb_send, 275.0, synth_cello_drone(9.5, 155.56, 155.56, 0.005, 1100, att=2.5, rel=2.0), gain=0.14)  # Eb3 gently fades out before open 5th!
    add_stereo(reverb_send, 274.5, synth_cello_drone(10.5, 98.00, 98.00, 0.004, 860, att=2.5, rel=1.5), gain=0.22)

    # Shot 25-26 (284-300s): Wind dies at 297.0s + pure haunting OPEN FIFTH (C2 + G2 + C3 + G3) decaying to 0 at 300.0s!
    add_stereo(amb, 284.0, synth_wind(13.0, low_hz=55, high_hz=300, gust_hz=0.12), gain=0.10)
    # Hard clean wind cut at 04:57.00 (297.0s)
    amb[int(297.0 * SR):] = 0.0
    # Open fifth C–G chord decaying gracefully to complete silence at 300.0s!
    add_stereo(reverb_send, 283.5, synth_cello_drone(16.2, 65.41, 65.41, 0.003, 620, att=1.2, rel=4.8), gain=0.30)
    add_stereo(reverb_send, 283.5, synth_cello_drone(16.2, 98.00, 98.00, 0.003, 760, att=1.4, rel=4.8), gain=0.24)
    add_stereo(reverb_send, 284.0, synth_cello_drone(15.6, 130.81, 130.81, 0.003, 880, att=1.6, rel=5.0), gain=0.18)
    add_stereo(reverb_send, 285.5, synth_cello_drone(14.0, 196.00, 196.00, 0.003, 1050, att=2.2, rel=5.0), gain=0.11)
    # 04:56.00 (296.0s): One faint metallic bolt-tightening click!
    add_mono(fol, 296.0, synth_click(14, 2400), gain=0.14)
    add_mono(fol, 296.0, synth_metal_clang(0.45, 520), gain=0.05)

    # Apply lush dark Arctic hall reverb to reverb_send bus
    print("Applying dark Arctic hall convolution reverb...")
    ir = make_dark_reverb_ir(3.2)
    rev_wet = apply_stereo_reverb(reverb_send, ir, wet=0.45)

    # Sum soundscape buses
    bed = amb * 0.72 + fol * 0.92 + sco * 0.85 + rev_wet * 0.88
    # Clean fade-out last 0.4s to absolute zero at 05:00.000
    tail_n = int(0.4 * SR)
    bed[-tail_n:] *= np.linspace(1, 0, tail_n, dtype=np.float32)[:, None]

    # Save audio cue metadata for video sync (relay clicks, moon cut, 180°C lock, snap cue, etc.)
    cues = {
        "shot03_cut_t": round(cut_t, 3),
        "shot04_well_t": round(well_t, 3),
        "shot06_moon_t0": round(moon_t0, 3),
        "shot06_moon_t1": round(moon_t1, 3),
        "shot07_down_t": round(down_t, 3),
        "shot11_180_start": round(t_180_start, 3),
        "shot11_180_lock": round(t_180_lock, 3),
        "shot15_nobasalt_t": round(nobasalt_t, 3),
        "shot17_h2_end": round(h2_end, 3),
        "shot19_snap_t": round(snap_cue, 3),
        "shot20_lock_t": round(lock_t, 3),
        "shot21_relay_times": relay_times,
    }
    with open(f"{AUDIO_DIR}/sync_cues.json", "w") as f:
        json.dump(cues, f, indent=2)

    # Load mastered narration & apply gentle smooth sidechain ducking (-4.8 dB during speech)
    sr_n, nar_int = wav.read(f"{AUDIO_DIR}/narration_master.wav")
    assert sr_n == SR
    nar = nar_int.astype(np.float32) / 32768.0
    if len(nar) < N:
        nar = np.pad(nar, ((0, N - len(nar)), (0, 0)))
    else:
        nar = nar[:N]

    # Smooth speech activity envelope (20ms RMS -> 350ms release smooth)
    nar_mono = np.abs(nar[:, 0])
    win = int(0.18 * SR)
    env = signal.fftconvolve(nar_mono, np.ones(win, dtype=np.float32) / win, mode="same")
    # Smooth sidechain gain: 1.0 in silence, 0.54 (-5.3 dB) during speech
    speech_mask = np.clip((env - 0.008) / 0.035, 0.0, 1.0)
    duck = 1.0 - 0.46 * speech_mask
    bed_ducked = bed * duck[:, None]

    # Balance: clear authoritative narration on top of rich atmospheric bed
    # Normalize bed RMS gently so it never overwhelms voice
    bed_peak = np.max(np.abs(bed_ducked)) + 1e-9
    bed_ducked = (bed_ducked / bed_peak) * 0.72
    mix = nar * 0.88 + bed_ducked * 0.62
    # Soft-clip safety limiter
    mix = np.tanh(mix * 0.95) * 0.92

    raw_mix = f"{AUDIO_DIR}/mix_pre_norm.wav"
    wav.write(raw_mix, SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))

    # Broadcast loudness master (-14 LUFS, -1.0 dBTP) via FFmpeg loudnorm
    master_out = f"{AUDIO_DIR}/audio_master.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_mix,
        "-af", "loudnorm=I=-14:TP=-1.0:LRA=11",
        "-ar", str(SR), "-ac", "2", "-t", f"{TOTAL_SEC:.3f}",
        master_out
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"DONE! Mastered 300.000s stereo audio -> {master_out}")
    print("Sync cues:", json.dumps(cues))

if __name__ == "__main__":
    main()
