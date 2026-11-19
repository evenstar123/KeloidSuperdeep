#!/usr/bin/env python3
"""Act 5 (Shots 19-26) frame renderers: Fracture, Abandonment, Scale & End Card."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from gfx_utils import (
    W, H, FPS, BG, IRON, WHITE, BLUE, GRID, AMBER, RED, RUST, STRATA,
    get_font, ease_out_cubic, ease_in_out_cubic, clamp01, lerp_color, fade_color,
    draw_tracked_text, draw_dashed_segment, draw_dashed_polyline,
    add_bloom, ken_burns, draw_hud_depth_ruler, draw_mg_grid, draw_snow_streaks,
)
from shots_act3_4 import _draw_temp_profile

# ---------- SHOT 19 (03:35 - 03:47 | MG 1984 Drill String Fracture at 12,066 m, 5,000 m Lost) ----------
def render_shot19(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.25)

    snap_local = ctx["cues"]["shot19_snap_t"] - 215.0  # ~5.08s!
    cx = 980
    y_top, y_bot = 140, 940  # represents 0 -> 13,000 m scale
    def depth_y(m): return y_top + (m / 13000.0) * (y_bot - y_top)

    y_7k = depth_y(7000.0)
    y_12066 = depth_y(12066.0)

    # Borehole casing channel walls
    d.rectangle([cx - 26, y_top, cx + 26, y_12066 + 18], fill=(16, 20, 26), outline=(52, 58, 66), width=1)
    d.text((cx - 120, y_top - 30), "SURFACE  0 m", font=get_font("mono_r", 14), fill=(135, 142, 150))
    d.line([(cx - 140, y_top), (cx + 140, y_top)], fill=WHITE, width=1)

    # Torsion mechanical vibration builds 1.5s before snap!
    if snap_local - 1.5 <= t < snap_local:
        vib_p = (t - (snap_local - 1.5)) / 1.5
        jx = int(math.sin(t * 95.0) * 6.5 * vib_p)
    else:
        jx = 0

    if t < snap_local:
        # Intact amber drill string down to 12,066 m, stressing near 7,000 m
        d.line([(cx + jx, y_top), (cx + jx, y_12066)], fill=AMBER, width=5)
        gd.line([(cx + jx, y_top), (cx + jx, y_12066)], fill=AMBER, width=5)
        d.ellipse([cx + jx - 6, y_12066 - 6, cx + jx + 6, y_12066 + 6], fill=WHITE)
        d.text((cx + 68, y_12066 - 24), "12,066 m", font=get_font("mono_b", 44), fill=AMBER)
        d.text((cx + 70, y_12066 + 26), "1984.09.27  ·  SG-3 MAIN SHAFT", font=get_font("mono_r", 16), fill=(165, 172, 180))
        # Stress warning glint building at 7,000 m right before break
        if t > snap_local - 1.3:
            sp = clamp01((t - (snap_local - 1.3)) / 1.3)
            d.ellipse([cx + jx - 10, y_7k - 10, cx + jx + 10, y_7k + 10], outline=fade_color(RED, sp), width=2)
            gd.ellipse([cx + jx - 12, y_7k - 12, cx + jx + 12, y_7k + 12], fill=fade_color(RED, sp * 0.8, (0, 0, 0)))
    else:
        # AFTER SNAP! Upper 7,000 m intact amber; lower 5,000 m turns ALERT RED, detaches & plunges into blackness!
        dt = t - snap_local
        # Upper intact string (recoil jitter decays fast)
        rec = int(math.sin(dt * 60.0) * 4.0 * math.exp(-dt * 8.0))
        d.line([(cx + rec, y_top), (cx + rec, y_7k)], fill=AMBER, width=4)
        gd.line([(cx + rec, y_top), (cx + rec, y_7k)], fill=fade_color(AMBER, 0.7, (0, 0, 0)), width=4)

        # Fracture flash right at break point!
        flash = clamp01(1.0 - dt / 0.35)
        if flash > 0.01:
            r_fl = int(12 + 52 * (1.0 - flash))
            d.ellipse([cx - r_fl, y_7k - r_fl, cx + r_fl, y_7k + r_fl], outline=fade_color(RED, flash), width=3)
            gd.ellipse([cx - 18, y_7k - 18, cx + 18, y_7k + 18], fill=fade_color(WHITE, flash, (0, 0, 0)))

        # Detached 5,000 m bottom section plunges downward & fades to #0A0A0C!
        drop_px = (dt ** 1.65) * 105.0
        fade_lost = clamp01(1.0 - (dt - 0.5) / 4.2)
        if fade_lost > 0.02:
            y0_lost = y_7k + 14 + drop_px
            y1_lost = min(H + 200, y_12066 + 14 + drop_px)
            c_red = fade_color(RED, fade_lost)
            d.line([(cx, y0_lost), (cx, y1_lost)], fill=c_red, width=5)
            gd.line([(cx, y0_lost), (cx, y1_lost)], fill=fade_color(RED, fade_lost * 0.85, (0, 0, 0)), width=5)

        # Crisp red callout box!
        al_c = ease_out_cubic(clamp01(dt / 0.6))
        bx, by = cx + 95, int(y_7k - 22)
        d.line([(cx + 12, y_7k), (bx - 10, y_7k)], fill=fade_color(RED, al_c), width=1)
        d.rectangle([bx, by - 16, bx + 530, by + 102], fill=BG, outline=fade_color(RED, al_c), width=2)
        d.text((bx + 24, by), "FRACTURE AT 7,000 m", font=get_font("mono_r", 17), fill=fade_color(WHITE, al_c * 0.85))
        d.text((bx + 24, by + 32), "STRING LOST: ~5,000 m", font=get_font("mono_b", 36), fill=fade_color(RED, al_c))
        gd.text((bx + 24, by + 32), "STRING LOST: ~5,000 m", font=get_font("mono_b", 36), fill=fade_color(RED, al_c * 0.55, (0, 0, 0)))

    draw_hud_depth_ruler(d, gd, 12066.0 if t < snap_local else 7000.0, 13000.0,
                         alpha=1.0, highlight_color=RED if t >= snap_local else AMBER)
    return add_bloom(im, glow, radius=6, strength=0.45)

# ---------- SHOT 20 (03:47 - 04:00 | MG Sidetrack from 7,000 m -> Final Lock 12,262 m) ----------
def render_shot20(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.25)

    lock_local = ctx["cues"]["shot20_lock_t"] - 227.0  # ~9.7s local!
    cx = 920
    y_top, y_bot = 135, 935
    def depth_y(m): return y_top + (m / 13000.0) * (y_bot - y_top)

    y_7k = depth_y(7000.0)
    y_12066 = depth_y(12066.0)
    y_12262 = depth_y(12262.0)

    # Surface line
    d.line([(cx - 180, y_top), (cx + 280, y_top)], fill=WHITE, width=1)
    d.text((cx - 175, y_top - 28), "SURFACE  0 m", font=get_font("mono_r", 14), fill=(135, 142, 150))

    # Main upper shaft (0 -> 7,000 m)
    d.line([(cx, y_top), (cx, y_7k)], fill=AMBER, width=4)
    gd.line([(cx, y_top), (cx, y_7k)], fill=fade_color(AMBER, 0.65, (0, 0, 0)), width=4)

    # Abandoned broken original shaft (7,000 -> 12,066 m) as dim ghost dashed grey!
    draw_dashed_segment(d, cx, y_7k, cx, y_12066, (72, 78, 86), width=2, dash=8, gap=8)
    d.text((cx - 210, y_12066 - 10), "1984 ABANDONED (12,066 m)", font=get_font("mono_r", 13), fill=(95, 102, 110))
    d.text((cx - 160, y_7k - 10), "SIDETRACK  7,000 m", font=get_font("mono_r", 14), fill=WHITE)
    d.ellipse([cx - 5, y_7k - 5, cx + 5, y_7k + 5], fill=WHITE)

    # New sidetrack branch splits gently rightward from 7,000 m down to 12,262 m!
    # Timeline: 0-2.5s cursor pulls back up to 7,000m; 2.5 -> lock_local drives down 7,000 -> 12,262 m!
    if t < 2.5:
        cur_depth = 12066.0 - (12066.0 - 7000.0) * ease_in_out_cubic(t / 2.5)
        branch_p = 0.0
    else:
        raw_p = clamp01((t - 2.5) / max(1.5, lock_local - 2.5))
        # Subtle 2-frame lock overshoot at 12,262 m!
        if raw_p < 1.0:
            branch_p = raw_p
        else:
            dt_l = t - lock_local
            branch_p = 1.0 + 0.008 * math.exp(-dt_l * 18.0) * math.sin(dt_l * 45.0)
        cur_depth = 7000.0 + (12262.0 - 7000.0) * min(1.0, raw_p)

    # Smooth curved sidetrack path
    n_seg = 60
    b_pts = []
    for i in range(n_seg + 1):
        u = i / n_seg
        bx = cx + 96.0 * (1.0 - math.exp(-u * 2.6))
        by = y_7k + (y_12262 - y_7k) * u
        b_pts.append((bx, by))

    k_b = int(round(clamp01(branch_p) * n_seg))
    if k_b >= 2:
        d.line(b_pts[:k_b + 1], fill=AMBER, width=4)
        gd.line(b_pts[:k_b + 1], fill=AMBER, width=4)
        tx, ty = b_pts[k_b]
        d.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=WHITE)
        gd.ellipse([tx - 9, ty - 9, tx + 9, ty + 9], fill=AMBER)

        # Big odometer readout tracking depth -> locks cleanly at 12,262 m!
        lbl = f"{int(round(cur_depth)):,} m"
        d.text((tx + 46, ty - 56), lbl, font=get_font("mono_b", 64), fill=AMBER)
        gd.text((tx + 46, ty - 56), lbl, font=get_font("mono_b", 64), fill=(210, 140, 0))
        if t >= lock_local:
            al_sub = ease_out_cubic(clamp01((t - lock_local) / 0.8))
            d.text((tx + 48, ty + 20), "SG-3 · TRUE VERTICAL DEPTH · 1989", font=get_font("mono_r", 18), fill=fade_color(WHITE, al_sub * 0.9))
            # Horizontal world-record hairline
            d.line([(cx - 120, y_12262), (tx + 40, y_12262)], fill=fade_color(AMBER, al_sub * 0.55), width=1)

    draw_hud_depth_ruler(d, gd, cur_depth, 13000.0, alpha=1.0, highlight_color=AMBER)
    return add_bloom(im, glow, radius=6, strength=0.40)

# ---------- SHOT 21 (04:00 - 04:12 | Hybrid MG 15km ≈300°C Extrapolation -> Real Soviet Control Room Lights Off) ----------
def render_shot21(t: float, dur: float, ctx: dict) -> Image.Image:
    if t < 5.0:
        # Part A (0-5s): Temperature profile zoomed out to 15 km / 320 °C, dashed red extrapolation locks ≈ 300 °C!
        im = Image.new("RGB", (W, H), BG)
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(im)
        gd = ImageDraw.Draw(glow)
        draw_mg_grid(d, 0.25)

        p_ext = ease_out_cubic(clamp01((t - 0.4) / 3.0))
        draw_hud_depth_ruler(d, gd, 12000.0 + 3000.0 * p_ext, 15000.0, alpha=1.0, highlight_color=RED)
        _draw_temp_profile(im, glow, d, gd, 1.0, 1.0, p_ext, max_depth_km=15.0, max_temp_c=320.0, show_delta=0.75)
        return add_bloom(im, glow, radius=6, strength=0.40)
    else:
        # Part B (5-12s): Authentic Soviet VVER Control Room (shot21b_ctrl), 8 indicator lamps switching OFF synced to relay clicks!
        abs_t = 240.0 + t
        relays = ctx["cues"]["shot21_relay_times"]  # 8 exact timestamps from 245.4 -> 249.95!
        # Static camera with darkening overall exposure as power drops!
        n_off = sum(1 for r in relays if abs_t >= r)
        base = ctx["plates"]["shot21b_ctrl"].crop((120, 68, 120 + W, 68 + H))
        arr = np.asarray(base, dtype=np.float32)
        # Room gets progressively darker as each panel circuit powers down!
        dim = 1.0 - 0.50 * (n_off / 8.0)
        arr *= dim
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        glow = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(im)
        gd = ImageDraw.Draw(glow)

        # 8 glowing analog Soviet console indicator lamps arrayed along the curved panel perspectives!
        lamp_coords = [
            (460, 445), (615, 432), (795, 420), (985, 408),
            (1185, 396), (1380, 384), (1565, 372), (1735, 360),
        ]
        for i, (lx, ly) in enumerate(lamp_coords):
            is_on = abs_t < relays[i]
            if is_on:
                col_l = AMBER if (i % 3 != 0) else (120, 210, 155)
                d.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=col_l, outline=WHITE, width=1)
                gd.ellipse([lx - 18, ly - 18, lx + 18, ly + 18], fill=col_l)
            else:
                # Extinguished dark glass indicator housing
                d.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=(22, 24, 28), outline=(58, 62, 68), width=1)

        # Subtle vintage December 1991 wall calendar plaque upper-left
        d.rectangle([92, 68, 290, 152], fill=(14, 16, 20), outline=(72, 78, 86), width=1)
        d.text((112, 80), "1991 · XII", font=get_font("mono_b", 26), fill=(195, 200, 206))
        d.text((112, 118), "USSR · DISSOLVED", font=get_font("mono_r", 13), fill=(130, 138, 146))

        return add_bloom(im, glow, radius=10, strength=0.65)

# ---------- SHOT 22 (04:12 - 04:20 | AI/Archival Real Kola Concrete Ruin, Static Tripod + Snow Dust) ----------
def render_shot22(t: float, dur: float, ctx: dict) -> Image.Image:
    base = ctx["plates"]["shot22_ruin"].copy()
    # Gentle drifting Arctic snow dust swirling silently across the abandoned concrete facility
    im = draw_snow_streaks(base, t, count=110, speed_x=-480.0, speed_y=65.0, streak_len=14, intensity=0.36, seed=2201)
    im = draw_snow_streaks(im, t + 2.3, count=70, speed_x=-220.0, speed_y=45.0, streak_len=3, intensity=0.30, seed=2202)
    return im

# ---------- SHOT 23 (04:20 - 04:31 | ICONIC REAL 12262 RUSTED BOLTED CAP + Ø 23 cm HUD) ----------
def render_shot23(t: float, dur: float, ctx: dict) -> Image.Image:
    prog = clamp01(t / dur)
    # Imperceptibly slow vertical push-in onto the actual bolted 12262 rusted metal borehole cap!
    im = ken_burns(ctx["plates"]["shot23_cap"], prog, z0=1.00, z1=1.07, cx0=0.50, cy0=0.50, cx1=0.50, cy1=0.50)
    # Delicate sparse falling snowflakes landing across the rusted iron
    im = draw_snow_streaks(im, t, count=48, speed_x=-65.0, speed_y=95.0, streak_len=2, intensity=0.38, seed=2301)

    d = ImageDraw.Draw(im)
    # Minimalist Permafrost White diameter caliper lines measuring the bolted cap: Ø 23 cm
    al = ease_out_cubic(clamp01((t - 2.6) / 1.0))
    if al > 0.01:
        # Caliper spans across the actual cap width centered in frame
        cx, cy = W // 2 - 20, H // 2 + 12
        half_cap = int(305 * (1.0 + 0.065 * prog))
        y_cal = cy - 235
        c_w = fade_color(WHITE, al * 0.85)
        c_dim = fade_color(WHITE, al * 0.42)
        # Vertical witness lines down to cap edges
        draw_dashed_segment(d, cx - half_cap, y_cal - 10, cx - half_cap, cy - 20, c_dim, 1, 6, 6)
        draw_dashed_segment(d, cx + half_cap, y_cal - 10, cx + half_cap, cy - 20, c_dim, 1, 6, 6)
        # Horizontal caliper bar with end ticks
        d.line([(cx - half_cap, y_cal), (cx + half_cap, y_cal)], fill=c_w, width=2)
        d.line([(cx - half_cap, y_cal - 10), (cx - half_cap, y_cal + 10)], fill=c_w, width=2)
        d.line([(cx + half_cap, y_cal - 10), (cx + half_cap, y_cal + 10)], fill=c_w, width=2)
        # Centered monospace label "Ø 23 cm" with subtle dark scrim
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(ov).rectangle([cx - 86, y_cal - 54, cx + 86, y_cal - 10], fill=(10, 10, 12, int(185 * al)))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(im)
        draw_tracked_text(d, cx, y_cal - 32, "Ø 23 cm", get_font("mono_b", 32), c_w, tracking_px=2)

    return im

# ---------- SHOT 24 (04:31 - 04:44 | MG Earth 6,371 km -> Rapid Zoom 0.19% -> Apple Skin Morph!) ----------
def render_shot24(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    cx = W // 2
    if t < 8.4:
        # Phase 1 (0 - 8.4s): Full Earth cross-section -> Exponential high-speed zoom into North Crust Surface!
        # Radius grows from 340 px (full Earth visible) -> 16,500 px (surface is nearly flat arc, 12.262 km = 32 px hairline!)
        zoom_p = ease_in_out_cubic(clamp01((t - 2.0) / 4.2))
        R_earth = 340.0 * (48.0 ** zoom_p)
        # Keep top surface of Earth positioned gracefully around y = 260..320 as we zoom in!
        y_surf_top = 180.0 + 160.0 * zoom_p
        cy = y_surf_top + R_earth

        # Draw Earth outer crust, mantle, outer core, inner core rings!
        fade_int = 1.0 - clamp01(zoom_p * 1.8)
        if fade_int > 0.01:
            # Inner core, outer core, lower mantle concentric wireframe rings
            for frac, col_a in [(0.19, 0.25), (0.54, 0.35), (0.985, 0.45)]:
                rr = R_earth * frac
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=fade_color(IRON, fade_int * col_a + 0.15), width=1)

        # Earth surface circumference arc (always razor sharp via sampled polyline when R is huge!)
        if R_earth < 2400:
            d.ellipse([cx - R_earth, cy - R_earth, cx + R_earth, cy + R_earth], outline=WHITE, width=2)
        else:
            arc_pts = []
            for px in range(0, W + 16, 16):
                dx = px - cx
                if abs(dx) < R_earth:
                    py = cy - math.sqrt(R_earth * R_earth - dx * dx)
                    if -50 < py < H + 50:
                        arc_pts.append((px, py))
            if len(arc_pts) >= 2:
                d.line(arc_pts, fill=WHITE, width=2)
                # Subtle mantle depth tint below crust
                poly = arc_pts + [(W, H), (0, H)]
                d.polygon(poly, fill=(14, 18, 24))
                d.line(arc_pts, fill=WHITE, width=2)

        # Radius line "6,371 km" visible during overview
        if fade_int > 0.02:
            c_rad = fade_color(WHITE, fade_int * 0.9)
            d.line([(cx, cy), (cx + R_earth, cy)], fill=c_rad, width=1)
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=c_rad)
            d.text((cx + 55, cy - 38), "R = 6,371 km", font=get_font("mono_b", 32), fill=c_rad)
            d.text((cx + 55, cy + 8), "EARTH MEAN RADIUS", font=get_font("mono_r", 15), fill=fade_color((140, 148, 156), fade_int))

        # The microscopic AMBER borehole hairline (true physical scale: 12.262 / 6371 = 0.0019246 * R_earth)!
        bore_px = max(2.0, R_earth * (12.262 / 6371.0))
        d.line([(cx, y_surf_top), (cx, y_surf_top + bore_px)], fill=AMBER, width=3)
        gd.line([(cx, y_surf_top), (cx, y_surf_top + bore_px)], fill=AMBER, width=4)

        # Callout fades in cleanly ABOVE the Earth arc (zero overlap with crust line!)
        al_z = ease_out_cubic(clamp01((t - 4.8) / 1.4))
        if al_z > 0.01:
            pct = 0.19 * ease_out_cubic(clamp01((t - 5.2) / 1.8))
            d.line([(cx, y_surf_top - 4), (cx, y_surf_top - 52)], fill=fade_color(AMBER, al_z * 0.85), width=1)
            d.line([(cx, y_surf_top - 52), (cx + 125, y_surf_top - 52)], fill=fade_color(AMBER, al_z * 0.85), width=1)
            d.text((cx + 140, y_surf_top - 84), "12.262 km", font=get_font("mono_b", 40), fill=fade_color(AMBER, al_z))
            gd.text((cx + 140, y_surf_top - 84), "12.262 km", font=get_font("mono_b", 40), fill=fade_color(AMBER, al_z * 0.6, (0, 0, 0)))
            d.text((cx + 140, y_surf_top - 34), f"{pct:.2f}%  OF EARTH RADIUS", font=get_font("mono_b", 22), fill=fade_color(WHITE, al_z * 0.92))
            # Big haunting centered 0.19% stat below
            draw_tracked_text(d, cx, 700, f"{pct:.2f}%", get_font("mono_b", 124), fade_color(AMBER, al_z), tracking_px=4)
            draw_tracked_text(gd, cx, 700, f"{pct:.2f}%", get_font("mono_b", 124), fade_color(AMBER, al_z * 0.5, (0, 0, 0)), tracking_px=4)
    else:
        # Phase 2 (8.4 - 13.0s): Minimalist 1px geometric Apple cross-section — borehole stops INSIDE the apple skin!
        tl = t - 8.4
        al_ap = ease_out_cubic(clamp01(tl / 0.8))
        acx, acy = 630, 560
        # Unmistakable minimalist geometric apple silhouette (clear top/bottom dimples, wider upper shoulders!)
        ap_pts = []
        for i in range(180):
            th = (i / 180.0) * 2 * math.pi
            # Heart/apple polar contour: clear top notch at th=-pi/2, softer bottom dimple at th=+pi/2
            top_notch = math.exp(-((math.sin(th) + 1.0) * 3.8)) * 58.0
            bot_notch = math.exp(-((1.0 - math.sin(th)) * 4.5)) * 28.0
            shoulder = 1.08 - 0.14 * math.sin(th)  # wider near upper shoulders, slightly tapered toward base
            rx = 248.0 * shoulder
            ry = 245.0
            x = acx + rx * math.cos(th)
            y = acy + ry * math.sin(th) + top_notch - bot_notch
            ap_pts.append((x, y))
        ap_pts.append(ap_pts[0])
        d.line(ap_pts, fill=fade_color(WHITE, al_ap * 0.92), width=2)
        # Subtle inner core/seeds minimalist wireframe
        d.ellipse([acx - 24, acy - 24, acx - 6, acy + 24], outline=fade_color((85, 92, 100), al_ap), width=1)
        d.ellipse([acx + 6, acy - 24, acx + 24, acy + 24], outline=fade_color((85, 92, 100), al_ap), width=1)
        # Stem rising cleanly out of top dimple
        d.line([(acx, acy - 188), (acx + 22, acy - 282)], fill=fade_color(WHITE, al_ap * 0.8), width=2)

        # Magnifier box on apple's upper-right skin rim
        mx, my = acx + 195, acy - 148
        d.rectangle([mx - 22, my - 22, mx + 22, my + 22], outline=fade_color(AMBER, al_ap), width=1)

        # Right side: Magnified Apple Skin Cross-Section Box (10x detail!)
        bx0, by0, bx1, by1 = 1045, 250, 1760, 830
        d.line([(mx + 22, my - 22), (bx0, by0)], fill=fade_color(IRON, al_ap * 0.8), width=1)
        d.line([(mx + 22, my + 22), (bx0, by1)], fill=fade_color(IRON, al_ap * 0.8), width=1)
        d.rectangle([bx0, by0, bx1, by1], fill=(13, 15, 19), outline=fade_color(WHITE, al_ap * 0.6), width=1)

        # Inside magnified box: Apple outer skin layer (~84px thick band at top of apple flesh!)
        y_outer = by0 + 195
        y_inner = by0 + 280  # ~0.4% of apple radius (apple skin thickness!)
        d.rectangle([bx0 + 1, y_outer, bx1 - 1, y_inner], fill=(34, 39, 48))
        d.line([(bx0, y_outer), (bx1, y_outer)], fill=fade_color(WHITE, al_ap), width=2)
        draw_dashed_segment(d, bx0, y_inner, bx1, y_inner, fade_color(WHITE, al_ap * 0.65), 1, 10, 8)
        d.text((bx0 + 28, y_outer + 30), "APPLE SKIN  (~0.4% RADIUS)", font=get_font("mono_r", 18), fill=fade_color((195, 202, 210), al_ap))
        d.text((bx0 + 28, y_inner + 36), "FLESH  /  MANTLE", font=get_font("mono_r", 18), fill=fade_color((130, 138, 146), al_ap))

        # The AMBER Kola borehole hairline STOPS HALFWAY INSIDE THE APPLE SKIN!
        hx = (bx0 + bx1) // 2 + 55
        pen_p = ease_out_cubic(clamp01((tl - 0.5) / 1.4))
        bore_h = 40.0 * pen_p
        d.line([(hx, y_outer), (hx, y_outer + bore_h)], fill=AMBER, width=4)
        gd.line([(hx, y_outer), (hx, y_outer + bore_h)], fill=AMBER, width=4)
        d.ellipse([hx - 4, y_outer + bore_h - 4, hx + 4, y_outer + bore_h + 4], fill=WHITE)
        d.text((hx + 22, y_outer - 68), "SG-3 · 0.19%", font=get_font("mono_b", 32), fill=fade_color(AMBER, al_ap))
        d.text((hx + 22, y_outer - 26), "STOPPED INSIDE THE SKIN", font=get_font("mono_r", 16), fill=fade_color(WHITE, al_ap * 0.88))

    return add_bloom(im, glow, radius=6, strength=0.38)

# ---------- SHOT 25 (04:44 - 04:54 | Hybrid Seamless Crane-Up from Arctic Horizon/Cap to Vast Polar-Night Starfield) ----------
def _build_polar_night_sky() -> Image.Image:
    """Create a breathtaking high-resolution 1920x1920 vertical polar-night panorama
    (bottom: bleak dark Arctic horizon + distant Kola concrete silhouette + tiny cap dot;
    top: vast crisp cold low-saturation starry sky & faint polar Milky Way band)."""
    PH = 1920
    yy, xx = np.mgrid[0:PH, 0:W].astype(np.float32)
    # Sky gradient: abyssal black (#0A0A0C) zenith -> deep polar-night blue (#16222E) near horizon (y ≈ 1680)
    hor_y = 1660.0
    sky_grad = np.clip((yy / hor_y) ** 2.2, 0.0, 1.0)
    canvas = np.zeros((PH, W, 3), dtype=np.float32)
    canvas[:, :, 0] = (10 + 14 * sky_grad) / 255.0
    canvas[:, :, 1] = (10 + 24 * sky_grad) / 255.0
    canvas[:, :, 2] = (12 + 38 * sky_grad) / 255.0

    # Faint cold diagonal Milky Way / polar star-dust band across upper sky
    band_d = ((xx - W * 0.48) * 0.55 + (yy - 680.0) * 0.83) / 290.0
    rng = np.random.default_rng(13000000)
    mw_noise = rng.normal(0.0, 1.0, (PH // 8, W // 8)).astype(np.float32)
    mw_img = Image.fromarray(((np.clip(mw_noise * 0.25 + 0.5, 0, 1)) * 255).astype(np.uint8)).resize((W, PH), Image.Resampling.BILINEAR)
    mw_tex = (np.asarray(mw_img, dtype=np.float32) / 255.0)
    mw = np.exp(-(band_d ** 2)) * (0.04 + 0.08 * mw_tex) * (yy < hor_y)
    canvas[:, :, 0] += mw * 0.72
    canvas[:, :, 1] += mw * 0.80
    canvas[:, :, 2] += mw * 0.95

    # Ground tundra silhouette below horizon (y >= hor_y)
    ridge = hor_y - 14.0 * np.sin(xx / 280.0) - 7.0 * np.cos(xx / 115.0) - 3.0 * np.sin(xx / 45.0)
    ground_mask = yy >= ridge
    canvas[ground_mask, 0] = 9 / 255.0
    canvas[ground_mask, 1] = 10 / 255.0
    canvas[ground_mask, 2] = 12 / 255.0

    im = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="RGB")
    d = ImageDraw.Draw(im)

    # 900 crisp low-saturation cold white/blue multi-magnitude stars!
    for _ in range(900):
        sx = rng.uniform(0, W)
        sy = rng.uniform(10, hor_y - 30)
        mag = rng.exponential(0.55)
        br = min(1.0, 0.20 + mag * 0.36)
        r = 1 if mag < 1.2 else (2 if mag < 2.4 else 2.5)
        tint = rng.choice([(217, 224, 232), (200, 214, 232), (225, 222, 215)])
        col = tuple(int(c * br) for c in tint)
        if r <= 1.2:
            d.point((sx, sy), fill=col)
        else:
            d.ellipse([sx - r * 0.6, sy - r * 0.6, sx + r * 0.6, sy + r * 0.6], fill=col)

    # Distant dark concrete silhouette of the abandoned Kola site on the right horizon
    d.rectangle([1260, int(hor_y - 46), 1375, int(hor_y + 12)], fill=(7, 8, 10))
    d.rectangle([1305, int(hor_y - 72), 1345, int(hor_y - 46)], fill=(7, 8, 10))
    # Subtle dark concrete slab & pinpoint muted dot of the borehole cap dead center at bottom!
    d.ellipse([W // 2 - 4, int(hor_y + 72) - 4, W // 2 + 4, int(hor_y + 72) + 4], fill=(42, 34, 30), outline=(88, 62, 48), width=1)
    return im

def render_shot25(t: float, dur: float, ctx: dict) -> Image.Image:
    if "polar_sky" not in ctx:
        ctx["polar_sky"] = _build_polar_night_sky()
    pano = ctx["polar_sky"]  # 1920 x 1920

    # Seamless slow crane-up: starts near ground/horizon (y0=840 -> horizon at bottom) & tilts upward into vast sky (y0=710 -> horizon pinned at very bottom edge!)
    prog = ease_in_out_cubic(clamp01(t / 8.5))
    y0 = int(round(840 - 150 * prog))
    im = pano.crop((0, y0, W, y0 + H))

    # Subtle gentle star twinkle on a few brightest stars
    d = ImageDraw.Draw(im)
    # Ultra-delicate vertical scale HUD on right margin: top 13,000,000,000 ly, bottom 12.262 km
    al_top = ease_out_cubic(clamp01((t - 1.2) / 1.4))
    al_bot = ease_out_cubic(clamp01((t - 4.8) / 1.4))
    rx = W - 115
    if al_top > 0.01:
        d.line([(rx, 92), (rx, 152)], fill=fade_color(WHITE, al_top * 0.45), width=1)
        d.ellipse([rx - 3, 89, rx + 3, 95], fill=fade_color(WHITE, al_top * 0.85))
        d.text((rx - 285, 80), "13,000,000,000 ly", font=get_font("mono_r", 22), fill=fade_color(WHITE, al_top * 0.72))
        d.text((rx - 222, 110), "OBSERVABLE HORIZON", font=get_font("mono_r", 12), fill=fade_color((145, 152, 160), al_top * 0.65))
    if al_bot > 0.01:
        d.line([(rx, H - 156), (rx, H - 96)], fill=fade_color(AMBER, al_bot * 0.55), width=1)
        d.ellipse([rx - 3, H - 99, rx + 3, H - 93], fill=fade_color(AMBER, al_bot * 0.9))
        d.text((rx - 175, H - 122), "12.262 km", font=get_font("mono_r", 22), fill=fade_color(AMBER, al_bot * 0.82))
        d.text((rx - 188, H - 94), "DEEPEST BOREHOLE", font=get_font("mono_r", 12), fill=fade_color((145, 152, 160), al_bot * 0.65))
    # Faint vertical hairline connecting heaven & earth
    if al_bot > 0.01:
        draw_dashed_segment(d, rx, 165, rx, H - 170, fade_color(GRID, al_bot * 0.9), 1, 6, 14)

    return im

# ---------- SHOT 26 (04:54 - 05:00 | ENDING END-CARD - EXACT PART 3.2 SPEC) ----------
def render_shot26(t: float, dur: float, ctx: dict) -> Image.Image:
    # 04:54.00 - 04:54.50 (t: 0.0 - 0.5s, 12 frames): Shot 25 starry sky darkens smoothly to pure #0A0A0C!
    if t < 0.50:
        sky = render_shot25(10.0, 10.0, ctx)
        fade_sky = 1.0 - (t / 0.50)
        arr = (np.asarray(sky, dtype=np.float32) * fade_sky + np.array(BG, dtype=np.float32) * (1.0 - fade_sky)).astype(np.uint8)
        im = Image.fromarray(arr)
    else:
        im = Image.new("RGB", (W, H), BG)

    # 04:57.00 - 05:00.00 (t >= 3.0s): Pure abyssal black 3-second resonance silence!
    if t >= 3.00:
        return Image.new("RGB", (W, H), BG)

    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    # Text fade in: 0.0 -> 0.50s (12 frames); static hold 0.50 -> 2.50s; fade out 2.50 -> 3.00s (12 frames)!
    al_in = clamp01(t / 0.50)
    al_out = 1.0 - clamp01((t - 2.50) / 0.50)
    al = al_in * al_out

    if al > 0.01:
        cx, cy = W / 2, H / 2
        # Single tiny amber pinpoint dot above text (abstract symbol of the 23 cm borehole mouth!)
        d.ellipse([cx - 4, cy - 118, cx + 4, cy - 110], fill=fade_color(AMBER, al))
        gd.ellipse([cx - 7, cy - 121, cx + 7, cy - 107], fill=fade_color(AMBER, al * 0.85, (0, 0, 0)))

        # Line 1: 12,262 m (Mono Bold 64px / 48pt, Permafrost White)
        draw_tracked_text(d, cx, cy - 36, "12,262 m", get_font("mono_b", 64), fade_color(WHITE, al), tracking_px=4)
        # Line 2: 科拉超深钻孔 SG-3 · 1970 — 1992 (Noto Sans SC 24px / 18pt, 60% opacity)
        draw_tracked_text(d, cx, cy + 36, "科拉超深钻孔  SG-3  ·  1970 — 1992", get_font("sans_r", 24), fade_color(WHITE, al * 0.62), tracking_px=4)
        # Line 3: 69°23′N 30°36′E (Mono 18px / 14pt, 40% opacity)
        draw_tracked_text(d, cx, cy + 82, "69°23′N  30°36′E", get_font("mono_r", 19), fade_color(WHITE, al * 0.42), tracking_px=3)

    return add_bloom(im, glow, radius=6, strength=0.45)
