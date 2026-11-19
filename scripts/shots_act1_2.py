#!/usr/bin/env python3
"""Act 1 (Shots 01-05) and Act 2 (Shots 06-09) frame renderers."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from gfx_utils import (
    W, H, FPS, BG, IRON, WHITE, BLUE, GRID, AMBER, RED, STRATA,
    get_font, ease_out_cubic, ease_in_out_cubic, clamp01, lerp_color, fade_color,
    draw_tracked_text, draw_dashed_segment, draw_dashed_polyline, draw_dashed_rect,
    add_bloom, ken_burns, draw_hud_depth_ruler, draw_mg_grid, draw_snow_streaks,
)

# ---------- SHOT 01 (00:00 - 00:03 | MG Live Waveform) ----------
def render_shot01(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    wave = ctx["hell_tape"]  # 48kHz mono array
    sr = 48000
    # Left-to-right writing progression across 0 -> 3.0s
    x_left, x_right = 120, W - 120
    cy = H // 2
    # Overload begins at t = 2.0s!
    overload = clamp01((t - 1.95) / 0.55)
    col = lerp_color(WHITE, AMBER, overload)
    # Draw span extends left -> right cleanly
    draw_frac = clamp01(t / 2.2)
    n_pts = max(2, int((x_right - x_left) * draw_frac / 2))
    # Subtle baseline guide
    d.line([(x_left, cy), (x_right, cy)], fill=(28, 30, 34), width=1)

    # Sample actual tape audio window up to current time t (stretched slightly after 2.0s)
    amp_scale = 135.0 + 210.0 * overload
    clip_lim = 148.0
    pts = []
    for i in range(n_pts):
        u = i / max(1, (x_right - x_left) // 2)
        x = x_left + i * 2
        # Organic speech/wail envelope (distinct quiet pauses & surging bursts like a real haunting tape!)
        burst = (0.28 + 0.72 * (0.5 + 0.5 * math.sin(u * 13.5 - 0.8)) ** 1.6) * (0.45 + 0.55 * abs(math.sin(u * 29.0 + 1.2)))
        t_audio = (u * 2.4) * (1.0 - 0.35 * overload) + 0.15 * t
        idx = int(t_audio * sr) % (len(wave) - 128)
        seg = wave[idx:idx + 96]
        val = float(seg[np.argmax(np.abs(seg))]) * (burst * (1.0 - 0.35 * overload) + 0.55 * overload)
        y_off = val * amp_scale
        # Hard flat-top clipping when overloaded!
        if overload > 0.15:
            y_off = max(-clip_lim, min(clip_lim, y_off))
        pts.append((x, cy - y_off))

    w_line = 2 if overload > 0.3 else 1
    if len(pts) >= 2:
        d.line(pts, fill=col, width=w_line)
        if overload > 0.05:
            gd.line(pts, fill=fade_color(AMBER, overload * 0.9, (0, 0, 0)), width=2)
            # Clipping flat-top ceiling reference hairlines
            c_lim = fade_color((110, 78, 12), overload * 0.6)
            draw_dashed_segment(d, x_left, cy - clip_lim, x_right, cy - clip_lim, c_lim, 1, 8, 12)
            draw_dashed_segment(d, x_left, cy + clip_lim, x_right, cy + clip_lim, c_lim, 1, 8, 12)

    return add_bloom(im, glow, radius=6, strength=0.45)

# ---------- SHOT 02 (00:03 - 00:15 | Hybrid AI CRT TV + VHS Static + HUD) ----------
def render_shot02(t: float, dur: float, ctx: dict) -> Image.Image:
    prog = clamp01(t / dur)
    # Slow 8% push-in over 12 seconds
    im = ken_burns(ctx["plates"]["shot02_crt"], prog, z0=1.00, z1=1.08, cx0=0.50, cy0=0.49, cx1=0.52, cy1=0.48)
    arr = np.asarray(im, dtype=np.float32)

    # Subtle CRT phosphor flicker + animated VHS tracking noise inside screen zone
    flicker = 1.0 + 0.025 * math.sin(t * 37.0) + 0.015 * math.sin(t * 11.3)
    arr *= flicker
    # CRT scanline texture & rolling VHS tracking bar across screen region (roughly x: 840..1260, y: 260..690)
    rng = np.random.default_rng(int(t * 24) + 100)
    sx0, sx1 = int(830 - prog * 18), int(1275 + prog * 18)
    sy0, sy1 = int(250 - prog * 12), int(690 + prog * 12)
    # Rolling tracking glitch line
    band_y = int(sy0 + ((t * 110.0) % (sy1 - sy0)))
    bh = rng.integers(3, 11)
    by1 = min(sy1, band_y + bh)
    noise_band = rng.uniform(0, 75, (by1 - band_y, sx1 - sx0, 1)).astype(np.float32)
    arr[band_y:by1, sx0:sx1] = np.clip(arr[band_y:by1, sx0:sx1] + noise_band, 0, 255)
    # Fine horizontal scanlines over screen
    arr[sy0:sy1:3, sx0:sx1] *= 0.88

    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    # Typewriter HUD in left dark negative space (matches dark room composition!)
    s1 = "1989 · RUMOR"
    s2 = "DEPTH CLAIMED: 14,400 m"
    s3 = "SOURCE: UNVERIFIED BROADCAST"
    n1 = int(clamp01((t - 0.6) / 1.2) * len(s1))
    n2 = int(clamp01((t - 2.6) / 2.2) * len(s2))
    n3 = int(clamp01((t - 6.0) / 2.0) * len(s3))

    hx, hy = 112, 168
    d.line([(hx, hy - 14), (hx + 340, hy - 14)], fill=IRON, width=1)
    if n1 > 0:
        d.text((hx, hy), s1[:n1], font=get_font("mono_b", 22), fill=WHITE)
    if n2 > 0:
        txt2 = s2[:n2]
        d.text((hx, hy + 42), txt2, font=get_font("mono_b", 28), fill=AMBER)
        gd.text((hx, hy + 42), txt2, font=get_font("mono_b", 28), fill=(180, 120, 0))
    if n3 > 0:
        d.text((hx, hy + 88), s3[:n3], font=get_font("mono_r", 15), fill=(135, 142, 150))

    # Subtle blinking cursor
    if int(t * 2.5) % 2 == 0 and t > 0.6:
        d.rectangle([hx, hy + 122, hx + 10, hy + 138], fill=AMBER)

    return add_bloom(im, glow, radius=5, strength=0.28)

# ---------- SHOT 03 (00:15 - 00:28 | MG Split Temp Comparison + Live FFT Spectrogram Cut) ----------
def render_shot03(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.35)

    abs_t = 15.0 + t
    cut_t = ctx["cues"]["shot03_cut_t"]  # ~22.3s exact audio cut!

    # Upper half: Left (Rumor 1,100 °C) vs Right (Measured 180 °C)
    # Center vertical divider
    d.line([(W // 2, 100), (W // 2, 510)], fill=GRID, width=1)

    # Left dashed white box
    lbox = (210, 140, 860, 460)
    draw_dashed_rect(d, lbox, fade_color(WHITE, 0.55), width=2, dash=12, gap=8)
    d.text((248, 168), "传言  /  RUMOR CLAIM", font=get_font("sans_r", 22), fill=(155, 162, 170))
    p_left = ease_out_cubic(clamp01(t / 1.8))
    val_l = int(round(1100 * p_left))
    strike = ease_out_cubic(clamp01((t - 3.2) / 0.55))
    c_lnum = lerp_color(WHITE, (95, 100, 108), strike)
    num_l_str = f"{val_l:,} °C"
    draw_tracked_text(d, 535, 315, num_l_str, get_font("mono_b", 88), c_lnum, tracking_px=2)

    # 1px clean horizontal strike-through across 1,100 °C!
    if strike > 0.01:
        sx0, sx1 = 300, 300 + int(470 * strike)
        d.line([(sx0, 315), (sx1, 315)], fill=WHITE, width=2)
        d.text((248, 412), "STATUS: UNSUBSTANTIATED", font=get_font("mono_r", 15), fill=fade_color((140, 146, 154), strike))

    # Right solid amber box (Measured 180 °C)
    rbox = (1060, 140, 1710, 460)
    p_right = ease_out_cubic(clamp01((t - 2.0) / 1.5))
    if t > 1.7:
        al_r = clamp01((t - 1.7) / 0.4)
        d.rectangle(rbox, outline=fade_color(AMBER, al_r), width=2)
        gd.rectangle(rbox, outline=fade_color(AMBER, al_r * 0.7, (0, 0, 0)), width=2)
        d.text((1098, 168), "实测  /  MEASURED BOREHOLE", font=get_font("sans_r", 22), fill=fade_color(AMBER, al_r * 0.9))
        val_r = int(round(180 * p_right))
        num_r_str = f"{val_r} °C"
        draw_tracked_text(d, 1385, 315, num_r_str, get_font("mono_b", 88), fade_color(AMBER, al_r), tracking_px=2)
        draw_tracked_text(gd, 1385, 315, num_r_str, get_font("mono_b", 88), fade_color(AMBER, al_r * 0.55, (0, 0, 0)), tracking_px=2)
        d.text((1098, 412), "DEPTH: 12,000 m  ·  SG-3 LOG", font=get_font("mono_r", 15), fill=fade_color(WHITE, al_r * 0.7))

    # Lower half: Audio Forensics Spectrogram
    sp_top, sp_bot = 555, 835
    sp_l, sp_r = 210, 1710
    d.line([(sp_l, sp_top), (sp_r, sp_top)], fill=GRID, width=1)
    d.line([(sp_l, sp_bot), (sp_r, sp_bot)], fill=IRON, width=1)
    d.text((sp_l, sp_top - 28), "AUDIO SPECTRUM ANALYSIS  ·  0.1 — 4.0 kHz", font=get_font("mono_r", 14), fill=(130, 136, 144))

    # 72 frequency bars driven by real FFT of reversed/stretched tape, snapping flat at cut_t!
    wave = ctx["hell_tape"]
    n_bars = 72
    bar_w = (sp_r - sp_l) / n_bars
    is_cut = abs_t >= cut_t
    if not is_cut:
        # Compute real FFT slice moving backwards (reversed playback!)
        pos = int(max(0, (5.5 - t * 0.62) * 48000)) % (len(wave) - 2048)
        chunk = wave[pos:pos + 2048] * np.hanning(2048)
        spec = np.abs(np.fft.rfft(chunk))[:256]
        spec = spec / (np.max(spec) + 1e-6)
        for i in range(n_bars):
            bi = int(i * 220 / n_bars)
            mag = float(np.mean(spec[bi:bi + 4]))
            # Slow down visual energy slightly as cut approaches
            bh = max(3, int((mag ** 0.72) * (sp_bot - sp_top - 30)))
            bx0 = sp_l + i * bar_w + 3
            bx1 = bx0 + bar_w - 6
            col_b = lerp_color((90, 98, 108), AMBER, clamp01(mag * 1.4))
            d.rectangle([bx0, sp_bot - bh, bx1, sp_bot], fill=col_b)
    else:
        # Clean dead flat zero line after exact digital cut click!
        flash = clamp01(1.0 - (abs_t - cut_t) / 0.25)
        if flash > 0:
            d.line([(sp_l, sp_bot - 1), (sp_r, sp_bot - 1)], fill=WHITE, width=3)
        d.text((sp_r - 190, sp_top + 14), "SIGNAL TERMINATED", font=get_font("mono_b", 15), fill=(160, 166, 174))

    # Three forensic tags typing out sequentially
    tags = [
        ("SOURCE: 1972 FILM SOUNDTRACK", 5.5),
        ("TIME-STRETCHED", 7.8),
        ("LAYERED REVERB", 9.6),
    ]
    tx = sp_l
    for idx_t, (label, t0) in enumerate(tags):
        if t >= t0:
            prog_t = clamp01((t - t0) / 1.1)
            sub = label[:max(1, int(prog_t * len(label)))]
            col_tag = AMBER if idx_t == 0 else WHITE
            d.rectangle([tx, 865, tx + 8, 886], fill=col_tag)
            d.text((tx + 20, 863), sub, font=get_font("mono_b", 20), fill=col_tag)
            tx += 520 if idx_t == 0 else 360

    return add_bloom(im, glow, radius=6, strength=0.35)

# ---------- SHOT 04 (00:28 - 00:36 | AI Ruin Blizzard Static Tripod) ----------
def render_shot04(t: float, dur: float, ctx: dict) -> Image.Image:
    # Absolute locked static tripod camera!
    base = ctx["plates"]["shot04_blizzard"].copy()
    # Two layers of fierce horizontal snow streaks sweeping right -> left
    im = draw_snow_streaks(base, t, count=150, speed_x=-1850.0, speed_y=120.0, streak_len=52, intensity=0.45, seed=401)
    im = draw_snow_streaks(im, t + 1.7, count=90, speed_x=-1100.0, speed_y=90.0, streak_len=24, intensity=0.28, seed=402)
    return im

# ---------- SHOT 05 (00:36 - 00:40 | OPENING TITLE - EXACT PART 3.1 SPEC) ----------
def render_shot05(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    # Global 15-frame fade-out starting at 00:39.375 (t = 3.375 -> 4.0)
    fade_out = 1.0 - clamp01((t - 3.375) / 0.625)
    # Subtle 1fps organic film gate flicker
    flicker = 0.96 if (int(t) % 2 == 1 and (t % 1.0) < 0.05) else 1.0

    # 00:36.00 (t=0): Main title "科拉超深钻孔" 6-frame (0.25s) linear fade-in, tracking tightens +600 -> +400 over 12 frames (0.5s)
    al_main = clamp01(t / 0.25) * fade_out * flicker
    p_track = ease_out_cubic(clamp01(t / 0.50))
    track_px = 46.0 - 14.0 * p_track  # generous cinema serif tracking tightening smoothly
    if al_main > 0.01:
        draw_tracked_text(
            d, W / 2, H / 2 - 42, "科拉超深钻孔",
            get_font("serif_b", 82), fade_color(WHITE, al_main), tracking_px=track_px
        )

    # 00:37.00 (t=1.0): Subtitle "通往地狱的井" 12-frame (0.5s) fade-in at 70% opacity
    if t >= 1.0:
        al_sub = clamp01((t - 1.0) / 0.50) * 0.72 * fade_out * flicker
        draw_tracked_text(
            d, W / 2, H / 2 + 54, "通往地狱的井",
            get_font("sans_r", 30), fade_color(WHITE, al_sub), tracking_px=24.0
        )

        # 1px amber vertical drill shaft extending steadily from under subtitle to bottom edge over 3.0s!
        p_line = clamp01((t - 1.0) / 2.85)
        y_start = H / 2 + 104
        y_end = y_start + (H - y_start) * p_line
        c_line = fade_color(AMBER, fade_out)
        d.line([(W // 2, y_start), (W // 2, y_end)], fill=c_line, width=2)
        gd.line([(W // 2, y_start), (W // 2, y_end)], fill=fade_color(AMBER, fade_out * 0.75, (0, 0, 0)), width=2)
        # Tiny bright drill tip dot
        d.ellipse([W // 2 - 2, y_end - 2, W // 2 + 2, y_end + 2], fill=fade_color(WHITE, fade_out))

    return add_bloom(im, glow, radius=6, strength=0.38)

# ---------- SHOT 06 (00:40 - 00:55 | Hybrid Real Kola Tower + 2.3s AI Lunar Bootprint Insert) ----------
def render_shot06(t: float, dur: float, ctx: dict) -> Image.Image:
    abs_t = 40.0 + t
    m0 = ctx["cues"]["shot06_moon_t0"]
    m1 = ctx["cues"]["shot06_moon_t1"]

    # Hard cut to dead-still archival lunar bootprint during "十个月前，阿姆斯特朗刚刚踏上月球"!
    if m0 <= abs_t <= m1:
        im = ctx["plates"]["shot06b_moon"].copy()
        d = ImageDraw.Draw(im)
        d.text((92, H - 96), "APOLLO 11 · SEA OF TRANQUILITY · 1969.07", font=get_font("mono_r", 15), fill=(160, 165, 170))
        return im

    prog = clamp01(t / dur)
    im = ken_burns(ctx["plates"]["shot06a_tower"], prog, z0=1.00, z1=1.09, cx0=0.50, cy0=0.52, cx1=0.53, cy1=0.49)
    # Gentle drifting Arctic snow mist
    im = draw_snow_streaks(im, t, count=55, speed_x=-260.0, speed_y=35.0, streak_len=4, intensity=0.26, seed=601)
    # Soft fade-in first 0.5s from title black
    if t < 0.5:
        arr = (np.asarray(im, dtype=np.float32) * (t / 0.5) + np.array(BG) * (1.0 - t / 0.5)).astype(np.uint8)
        im = Image.fromarray(arr)

    d = ImageDraw.Draw(im)
    al_hud = ease_out_cubic(clamp01((t - 1.2) / 0.7))
    if al_hud > 0.01:
        hx, hy = 92, H - 126
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rectangle([hx - 16, hy - 22, hx + 332, hy + 56], fill=(10, 10, 12, int(175 * al_hud)))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(im)
        d.line([(hx, hy - 12), (hx + 312, hy - 12)], fill=fade_color(AMBER, al_hud * 0.85), width=1)
        d.text((hx, hy), "69°23′N  30°36′E  ·  1970.05", font=get_font("mono_b", 18), fill=fade_color(WHITE, al_hud * 0.96))
        d.text((hx, hy + 26), "KOLA PENINSULA · ARCTIC CIRCLE", font=get_font("mono_r", 13), fill=fade_color((175, 182, 190), al_hud))
    return im

# ---------- SHOT 07 (00:55 - 01:06 | MG 384,400 km UP -> Rotate 180° -> -15,000 m DOWN) ----------
def render_shot07(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.35)

    abs_t = 55.0 + t
    down_t = ctx["cues"]["shot07_down_t"]  # ~61.6s when "向下" hits!
    t_switch = down_t - 55.0               # local time ~6.6s

    cx = W // 2 + 60
    if t < t_switch - 0.8:
        # Phase 1: Horizontal surface line + WHITE vector rising UP to 384,400 km
        y_surf = 840
        p_up = ease_out_cubic(clamp01(t / 2.2))
        fade_p1 = 1.0 - clamp01((t - (t_switch - 1.6)) / 0.7)
        d.line([(260, y_surf), (W - 140, y_surf)], fill=fade_color(WHITE, 0.45 * fade_p1), width=1)
        d.text((W - 290, y_surf + 12), "SURFACE  0 m", font=get_font("mono_r", 14), fill=fade_color((130, 136, 144), fade_p1))

        y_top = y_surf - 660 * p_up
        c_w = fade_color(WHITE, fade_p1)
        d.line([(cx, y_surf), (cx, y_top)], fill=c_w, width=2)
        d.ellipse([cx - 5, y_top - 5, cx + 5, y_top + 5], fill=c_w)
        # Arrowhead pointing up
        d.polygon([(cx, y_top - 12), (cx - 6, y_top), (cx + 6, y_top)], fill=c_w)

        km_val = int(round(384400 * p_up))
        d.text((cx + 32, y_top - 22), f"+{km_val:,} km", font=get_font("mono_b", 54), fill=c_w)
        d.text((cx + 34, y_top + 44), "APOLLO 11 · EARTH — MOON", font=get_font("mono_r", 17), fill=fade_color((145, 152, 160), fade_p1))
    else:
        # Phase 2: Coordinate flip (0.8s smooth rotation) -> AMBER vector plunging DOWN to -15,000 m!
        rot_p = ease_in_out_cubic(clamp01((t - (t_switch - 0.8)) / 0.85))
        y_surf = int(840 - (840 - 165) * rot_p)
        d.line([(260, y_surf), (W - 140, y_surf)], fill=fade_color(WHITE, 0.55), width=1)
        d.text((W - 310, y_surf - 26), "SURFACE  0 m", font=get_font("mono_r", 14), fill=(135, 142, 150))

        # Plunge down starting right at t_switch!
        dive_raw = clamp01((t - t_switch) / 2.25)
        # 2-frame slight overshoot when locking at 15,000m (Part 1.2 spec!)
        if dive_raw < 1.0:
            p_dn = dive_raw
        else:
            dt_over = t - (t_switch + 2.25)
            p_dn = 1.0 + 0.015 * math.exp(-dt_over * 16.0) * math.sin(dt_over * 38.0)

        depth_m = min(15000.0, 15000.0 * max(0.0, dive_raw))
        y_bot = y_surf + 725 * max(0.0, p_dn)

        # Left HUD depth ruler appears smoothly!
        draw_hud_depth_ruler(d, gd, depth_m, 15000.0, alpha=rot_p)

        if t >= t_switch:
            d.line([(cx, y_surf), (cx, y_bot)], fill=AMBER, width=3)
            gd.line([(cx, y_surf), (cx, y_bot)], fill=AMBER, width=3)
            d.polygon([(cx, y_bot + 12), (cx - 7, y_bot), (cx + 7, y_bot)], fill=AMBER)

            # Blink twice after locking!
            blink_on = True
            if dive_raw >= 1.0:
                dt_lock = t - (t_switch + 2.25)
                if (0.12 < dt_lock < 0.24) or (0.38 < dt_lock < 0.50):
                    blink_on = False
            if blink_on:
                lbl = f"-{int(round(depth_m)):,} m"
                d.text((cx + 34, y_bot - 42), lbl, font=get_font("mono_b", 64), fill=AMBER)
                gd.text((cx + 34, y_bot - 42), lbl, font=get_font("mono_b", 64), fill=(200, 135, 0))
            d.text((cx + 36, y_bot + 32), "TARGET DEPTH · SOVIET PROJECT SG-3", font=get_font("mono_r", 17), fill=WHITE)

    return add_bloom(im, glow, radius=6, strength=0.35)

# ---------- SHOT 08 (01:06 - 01:22 | Hybrid MG Earth Moho Seismic Refraction -> Real Mohole Ship + RED STAMP) ----------
def render_shot08(t: float, dur: float, ctx: dict) -> Image.Image:
    if t < 10.0:
        # Part A (0-10s): Earth crust/mantle zoom + Moho amber dashed boundary + seismic P-wave refraction!
        im = Image.new("RGB", (W, H), BG)
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(im)
        gd = ImageDraw.Draw(glow)
        draw_mg_grid(d, 0.30)

        zoom = ease_in_out_cubic(clamp01(t / 5.0))
        # Stratified crustal slab moving gently upward as we zoom closer
        y_surf = int(230 - 50 * zoom)
        y_moho = int(560 - 20 * zoom)
        # Crust fill (#1E2A35) & Mantle fill (#16222E)
        d.rectangle([280, y_surf, W - 120, y_moho], fill=STRATA[1])
        d.rectangle([280, y_moho, W - 120, 920], fill=STRATA[0])
        d.line([(280, y_surf), (W - 120, y_surf)], fill=WHITE, width=2)
        d.text((305, y_surf + 20), "CONTINENTAL CRUST  /  地壳", font=get_font("sans_r", 22), fill=(175, 182, 190))
        d.text((305, y_moho + 28), "UPPER MANTLE  /  上地幔", font=get_font("sans_r", 22), fill=(140, 150, 160))

        # Moho discontinuity: AMBER dashed boundary (the visual focus!)
        draw_dashed_segment(d, 280, y_moho, W - 120, y_moho, AMBER, width=3, dash=18, gap=10)
        draw_dashed_segment(gd, 280, y_moho, W - 120, y_moho, AMBER, width=3, dash=18, gap=10)
        d.text((W - 490, y_moho - 36), "MOHO DISCONTINUITY · 莫霍面", font=get_font("sans_b", 22), fill=AMBER)

        # Animated seismic P-wave ray propagating from surface, refracting & accelerating at Moho!
        ray_p = clamp01((t - 1.2) / 4.2)
        rx0, ry0 = 560, y_surf
        rx1, ry1 = 920, y_moho
        rx2, ry2 = 1580, 725  # shallower refracted angle = higher velocity!
        if ray_p > 0:
            p1 = clamp01(ray_p * 2.0)
            px1 = rx0 + (rx1 - rx0) * p1
            py1 = ry0 + (ry1 - ry0) * p1
            d.line([(rx0, ry0), (px1, py1)], fill=WHITE, width=2)
            # Expanding wavefront arcs along ray 1
            for k in range(3):
                r_arc = ((t * 140 + k * 70) % 210)
                if r_arc < math.hypot(px1 - rx0, py1 - ry0):
                    d.arc([rx0 - r_arc, ry0 - r_arc, rx0 + r_arc, ry0 + r_arc], 15, 75, fill=(100, 112, 125), width=1)
            if ray_p > 0.5:
                p2 = clamp01((ray_p - 0.5) * 2.0)
                px2 = rx1 + (rx2 - rx1) * p2
                py2 = ry1 + (ry2 - ry1) * p2
                d.line([(rx1, ry1), (px2, py2)], fill=AMBER, width=3)
                gd.line([(rx1, ry1), (px2, py2)], fill=AMBER, width=3)
                d.ellipse([rx1 - 6, ry1 - 6, rx1 + 6, ry1 + 6], fill=AMBER)
                # Velocity jump callout box
                al_v = ease_out_cubic(p2)
                bx, by = 1040, y_moho + 140
                d.rectangle([bx, by, bx + 420, by + 94], fill=BG, outline=fade_color(AMBER, al_v), width=1)
                d.text((bx + 22, by + 14), "SEISMIC P-WAVE ACCELERATION", font=get_font("mono_r", 14), fill=fade_color((150, 158, 166), al_v))
                d.text((bx + 22, by + 42), "Vp 6.5 → 8.0 km/s", font=get_font("mono_b", 32), fill=fade_color(AMBER, al_v))

        # Left HUD ruler showing target scale (0-35 km crustal scale)
        draw_hud_depth_ruler(d, gd, 15000 * clamp01(t / 6.0), 15000.0, alpha=0.85)
        return add_bloom(im, glow, radius=6, strength=0.35)
    else:
        # Part B (10-16s): Vintage 16mm newsreel Project Mohole ocean drilling ship + ALERT RED STAMP!
        tl = t - 10.0
        # Gentle ocean swell roll & subtle zoom
        cy_roll = 0.50 + 0.012 * math.sin(tl * 1.4)
        im = ken_burns(ctx["plates"]["shot08b_mohole"], tl / 6.0, z0=1.02, z1=1.09, cx0=0.50, cy0=cy_roll, cx1=0.50, cy1=cy_roll)
        # Vintage 16mm flicker
        arr = np.asarray(im, dtype=np.float32) * (0.95 + 0.05 * math.sin(tl * 43.0))
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(im)
        gd = ImageDraw.Draw(glow)

        d.text((92, 68), "U.S. NATIONAL SCIENCE FOUNDATION · ARCHIVE 1961—1966", font=get_font("mono_r", 16), fill=(180, 185, 190))

        # ALERT RED STAMP slams dead center at tl = 3.4s (01:19.4, synced to sub-thud!)
        if tl >= 3.4:
            p_st = ease_out_cubic(clamp01((tl - 3.4) / 0.16))
            # Subtle impact shake first 0.15s
            shake_y = int(6 * math.exp(-(tl - 3.4) * 28.0) * math.sin((tl - 3.4) * 90.0))
            bw, bh = 860, 128
            x0, y0 = W // 2 - bw // 2, H // 2 - bh // 2 + shake_y
            # Dark translucent backing plate for razor legibility
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            od.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(10, 10, 12, int(215 * p_st)), outline=(*RED, int(255 * p_st)), width=4)
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            d = ImageDraw.Draw(im)
            stamp = "PROJECT MOHOLE · 1966 · CANCELLED"
            draw_tracked_text(d, W / 2, H / 2 + shake_y, stamp, get_font("mono_b", 34), fade_color(RED, p_st), tracking_px=2)
            draw_tracked_text(gd, W / 2, H / 2 + shake_y, stamp, get_font("mono_b", 34), fade_color(RED, p_st * 0.65, (0, 0, 0)), tracking_px=2)
        return add_bloom(im, glow, radius=6, strength=0.35)

# ---------- SHOT 09 (01:22 - 01:35 | Industrial Rig Floor Tongs -> Top-Down Rotary Borehole Mouth) ----------
def render_shot09(t: float, dur: float, ctx: dict) -> Image.Image:
    if t < 6.5:
        # Cut 1: Dark industrial rig tongs + sweeping amber warning beacon!
        prog = t / 6.5
        im = ken_burns(ctx["plates"]["shot09a_tongs"], prog, z0=1.02, z1=1.14, cx0=0.52, cy0=0.55, cx1=0.54, cy1=0.58)
        arr = np.asarray(im, dtype=np.float32)
        # Rotating amber warning lamp sweep across the steel rig floor!
        sweep_x = W * (0.15 + 0.70 * (0.5 + 0.5 * math.sin(t * 2.4)))
        yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
        beam = np.exp(-(((xx * 4 - sweep_x) / 360.0) ** 2 + ((yy * 4 - H * 0.48) / 420.0) ** 2))
        beam_full = np.repeat(np.repeat(beam, 4, axis=0), 4, axis=1)
        # Apply warm amber glint proportional to specular steel luminance
        lum = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / (3.0 * 255.0)
        glint = beam_full * np.clip(lum * 2.2, 0.05, 1.0) * 85.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] + glint * 1.0, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] + glint * 0.68, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + glint * 0.06, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    else:
        # Cut 2 (6.5 - 13s): Dramatic vertical camera descending over heavy machined steel rotary table & pipe going into blackness!
        tl = t - 6.5
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        cx, cy = W // 2, int(H * 0.55 + tl * 8.0)

        # Cold overhead industrial spotlight over steel rig deck floor plates!
        for r in range(780, 140, -16):
            sh = int(24 + 62 * (1.0 - (r - 140) / 640.0) ** 0.85)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(sh, sh + 4, sh + 9))
        # Deck steel grating seams
        for gx in range(cx - 700, cx + 701, 140):
            d.line([(gx, 0), (gx, H)], fill=(20, 23, 27), width=2)
        for gy in range(cy - 600, cy + 601, 140):
            d.line([(0, gy), (W, gy)], fill=(20, 23, 27), width=2)

        # Heavy rotary table machined concentric steel rings (high-contrast industrial iron!)
        d.ellipse([cx - 500, cy - 500, cx + 500, cy + 500], fill=(56, 62, 69), outline=(145, 154, 164), width=4)
        for rad, col, w in [(465, (30, 34, 39), 12), (390, (125, 134, 144), 4),
                            (315, (36, 40, 46), 22), (225, (152, 160, 170), 4), (168, (18, 20, 24), 18)]:
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=w)
        # Rotating rotary-table locking lugs & hexagonal bolts!
        rot_ang = tl * 1.45
        for k in range(12):
            a = rot_ang + k * (2 * math.pi / 12)
            bx = cx + 426 * math.cos(a)
            by = cy + 426 * math.sin(a)
            d.ellipse([bx - 18, by - 18, bx + 18, by + 18], fill=(42, 47, 54), outline=(158, 166, 176), width=2)
            d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(170, 178, 186))
        for k in range(4):
            a = rot_ang + k * (math.pi / 2)
            x1, y1 = cx + 172 * math.cos(a), cy + 172 * math.sin(a)
            x2, y2 = cx + 382 * math.cos(a), cy + 382 * math.sin(a)
            d.line([(x1, y1), (x2, y2)], fill=(78, 86, 95), width=22)
            d.line([(x1, y1), (x2, y2)], fill=(140, 148, 158), width=4)

        # Pitch-black central borehole aperture!
        d.ellipse([cx - 150, cy - 150, cx + 150, cy + 150], fill=(4, 4, 6), outline=(128, 136, 146), width=4)

        # Massive gleaming steel drill pipe descending continuously straight into the pitch-black aperture!
        pw = 112
        d.rectangle([cx - pw // 2, 0, cx + pw // 2, cy + 24], fill=(64, 70, 78))
        # High-contrast cylindrical chrome/oiled-steel shading stripes down the pipe!
        for dx, c_s in [(-48, (30, 34, 40)), (-30, (78, 86, 95)), (-12, (168, 178, 188)),
                        (2, (205, 212, 220)), (16, (108, 116, 126)), (34, (46, 52, 60)), (48, (22, 25, 30))]:
            d.rectangle([cx + dx - 8, 0, cx + dx + 8, cy + 20], fill=c_s)
        # Moving threaded pipe joints & tool-joint collars descending smoothly into the black hole!
        joint_spacing = 240
        offset_y = (tl * 105.0) % joint_spacing
        for jy in range(-240, cy + 240, joint_spacing):
            yy = jy + offset_y
            if -60 < yy < cy + 12:
                d.rectangle([cx - pw // 2 - 14, yy - 24, cx + pw // 2 + 14, yy + 24], fill=(82, 90, 99), outline=(185, 194, 204), width=2)
                for ty in range(int(yy - 18), int(yy + 20), 5):
                    d.line([(cx - pw // 2 - 12, ty), (cx + pw // 2 + 12, ty + 2)], fill=(28, 32, 38), width=2)
        # Shadow collar where pipe enters dark borehole void
        d.arc([cx - 150, cy - 150, cx + 150, cy + 150], 0, 180, fill=(6, 6, 8), width=36)
        d.ellipse([cx - pw // 2, cy + 14, cx + pw // 2, cy + 38], fill=(10, 11, 14))

        # Rotating warm amber warning beacon sweeping vividly across the wet steel rotary table & pipe!
        arr = np.asarray(im, dtype=np.float32)
        ang_b = tl * 2.8
        bx_g = cx + 360 * math.cos(ang_b)
        by_g = cy + 230 * math.sin(ang_b)
        yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
        beam = np.exp(-(((xx * 4 - bx_g) / 380.0) ** 2 + ((yy * 4 - by_g) / 340.0) ** 2))
        beam_f = np.repeat(np.repeat(beam, 4, axis=0), 4, axis=1)
        lum = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / (3.0 * 255.0)
        gl = beam_f * np.clip(lum * 2.5 + 0.08, 0.0, 1.0) * 140.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] + gl * 1.0, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] + gl * 0.68, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + gl * 0.04, 0, 255)
        # Clean edge vignette keeping center crisp & illuminated
        dx_v = (xx * 4 - W * 0.5) / (W * 0.62)
        dy_v = (yy * 4 - H * 0.5) / (H * 0.62)
        vig = np.clip(1.0 - 0.58 * (dx_v ** 2 + dy_v ** 2), 0.14, 1.0)
        vig_f = np.repeat(np.repeat(vig, 4, axis=0), 4, axis=1)
        arr = arr * vig_f[:, :, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
