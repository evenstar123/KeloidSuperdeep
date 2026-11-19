#!/usr/bin/env python3
"""Act 3 (Shots 10-13) and Act 4 (Shots 14-18) frame renderers."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from gfx_utils import (
    W, H, FPS, BG, IRON, WHITE, BLUE, GRID, AMBER, RED, STRATA,
    get_font, ease_out_cubic, ease_in_out_cubic, clamp01, lerp_color, fade_color,
    draw_tracked_text, draw_dashed_segment, draw_dashed_polyline,
    add_bloom, ken_burns, draw_hud_depth_ruler, draw_mg_grid, draw_fact_badge,
)

# ---------- SHOT 10 (01:35 - 01:46 | MG Depth vs Hardness Intuition Curve Dissolving) ----------
def render_shot10(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    draw_mg_grid(d, 0.30)

    ox, oy = 380, 185
    ax_w, ax_h = 1180, 690
    # Crisp axes
    d.line([(ox, oy), (ox + ax_w, oy)], fill=(125, 132, 140), width=2)  # X: Hardness
    d.line([(ox, oy), (ox, oy + ax_h)], fill=(125, 132, 140), width=2)  # Y: Depth (downward)
    d.polygon([(ox + ax_w + 12, oy), (ox + ax_w, oy - 6), (ox + ax_w, oy + 6)], fill=(125, 132, 140))
    d.polygon([(ox, oy + ax_h + 12), (ox - 6, oy + ax_h), (ox + 6, oy + ax_h)], fill=(125, 132, 140))

    d.text((ox + ax_w - 230, oy - 38), "ROCK HARDNESS  /  硬度", font=get_font("sans_r", 22), fill=WHITE)
    d.text((ox - 160, oy + ax_h - 24), "DEPTH / 深度", font=get_font("sans_r", 20), fill=WHITE)
    d.text((ox - 34, oy - 28), "0", font=get_font("mono_r", 16), fill=(130, 136, 144))

    # Build smooth diagonal intuition curve (0 -> 1)
    n_pts = 80
    all_pts = []
    for i in range(n_pts + 1):
        u = i / n_pts
        px = ox + u * (ax_w * 0.86)
        py = oy + (u ** 0.92) * (ax_h * 0.88)
        all_pts.append((px, py))

    # 0.4s -> 4.2s: draws forward smoothly; 5.6s -> 8.2s: dissolves BACKWARD from tip to origin!
    if t < 5.6:
        frac = ease_out_cubic(clamp01((t - 0.4) / 3.6))
    else:
        frac = 1.0 - ease_in_out_cubic(clamp01((t - 5.6) / 2.4))

    k = int(round(frac * n_pts))
    if k >= 2:
        al_c = clamp01(frac * 1.4)
        draw_dashed_polyline(d, all_pts[:k + 1], fade_color(WHITE, al_c), width=2, dash=14, gap=9)
        tx, ty = all_pts[k]
        d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=fade_color(WHITE, al_c))
        if t < 6.4:
            al_lbl = clamp01((t - 1.4) / 0.8) * (1.0 - clamp01((t - 5.4) / 0.8))
            d.text((tx + 20, ty - 14), "ASSUMED MODEL / 常规直觉", font=get_font("sans_r", 20), fill=fade_color((170, 178, 186), al_lbl))

    return im

# ---------- SHARED TEMPERATURE PROFILE CURVE HELPER (Shot 11 & Shot 21a) ----------
def _draw_temp_profile(im, glow, d, gd, prog_pred: float, prog_real: float, prog_extrap: float = 0.0,
                       max_depth_km: float = 12.0, max_temp_c: float = 200.0, show_delta: float = 0.0):
    ox, oy = 420, 155
    gw, gh = 1320, 745
    # Subtle inner gridlines
    for tk in range(0, int(max_temp_c) + 1, 50):
        x = ox + (tk / max_temp_c) * gw
        d.line([(x, oy), (x, oy + gh)], fill=(26, 29, 34), width=1)
        d.text((x - 24, oy - 34), f"{tk} °C", font=get_font("mono_r", 15), fill=(135, 142, 150))
    for dk in range(0, int(max_depth_km) + 1, 3):
        y = oy + (dk / max_depth_km) * gh
        d.line([(ox, y), (ox + gw, y)], fill=(26, 29, 34), width=1)
        d.text((ox - 68, y - 9), f"{dk:2d} km", font=get_font("mono_r", 15), fill=(135, 142, 150))

    # Main axes
    d.line([(ox, oy), (ox + gw, oy)], fill=(135, 142, 150), width=2)
    d.line([(ox, oy), (ox, oy + gh)], fill=(135, 142, 150), width=2)
    d.text((ox + gw - 255, oy - 64), "TEMPERATURE  /  井底温度", font=get_font("sans_r", 20), fill=WHITE)

    # Prediction model: linear 0 km (0°C) -> 12 km (100°C)
    # Measured curve: follows close up to 3 km, then bends steeply to 12 km (180°C)!
    def pred_temp(dk):
        return 5.0 + (95.0 / 12.0) * dk

    def real_temp(dk):
        if dk <= 3.0:
            return 5.0 + (28.0 / 3.0) * dk
        u = (dk - 3.0) / 9.0
        return 33.0 + 147.0 * (u ** 1.28)

    n_samp = 120
    pred_pts = []
    real_pts = []
    for i in range(n_samp + 1):
        dk = (i / n_samp) * 12.0
        px = ox + (pred_temp(dk) / max_temp_c) * gw
        py = oy + (dk / max_depth_km) * gh
        pred_pts.append((px, py))
        rx = ox + (real_temp(dk) / max_temp_c) * gw
        real_pts.append((rx, py))

    # Optional 10% translucent Alert Red delta fill between predicted & measured!
    if show_delta > 0.01 and prog_real > 0.25:
        k_fill = int(round(clamp01(prog_real) * n_samp))
        if k_fill >= 4:
            poly = pred_pts[:k_fill + 1] + list(reversed(real_pts[:k_fill + 1]))
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            od.polygon(poly, fill=(*RED, int(34 * show_delta)))
            im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))
            d = ImageDraw.Draw(im)

    # Draw predicted dashed white curve
    kp = int(round(clamp01(prog_pred) * n_samp))
    if kp >= 2:
        draw_dashed_polyline(d, pred_pts[:kp + 1], WHITE, width=2, dash=12, gap=8)
        px_e, py_e = pred_pts[kp]
        d.ellipse([px_e - 4, py_e - 4, px_e + 4, py_e + 4], fill=WHITE)
        cur_pt = int(round(pred_temp(12.0 * kp / n_samp)))
        d.text((px_e - 120, py_e + 16), f"PREDICTED  {cur_pt} °C", font=get_font("mono_r", 17), fill=(175, 182, 190))

    # Draw measured solid curve transitioning AMBER -> ALERT RED!
    kr = int(round(clamp01(prog_real) * n_samp))
    if kr >= 2:
        for i in range(kr):
            u = i / n_samp
            col = lerp_color(AMBER, RED, clamp01((u - 0.30) / 0.65))
            d.line([real_pts[i], real_pts[i + 1]], fill=col, width=3)
            gd.line([real_pts[i], real_pts[i + 1]], fill=col, width=3)
        rx_e, ry_e = real_pts[kr]
        col_end = lerp_color(AMBER, RED, clamp01((kr / n_samp - 0.30) / 0.65))
        d.ellipse([rx_e - 7, ry_e - 7, rx_e + 7, ry_e + 7], fill=col_end)
        gd.ellipse([rx_e - 9, ry_e - 9, rx_e + 9, ry_e + 9], fill=col_end)
        cur_rt = int(round(real_temp(12.0 * kr / n_samp)))
        lbl_r = f"{cur_rt} °C"
        d.text((rx_e + 22, ry_e - 64), lbl_r, font=get_font("mono_b", 48), fill=col_end)
        gd.text((rx_e + 22, ry_e - 64), lbl_r, font=get_font("mono_b", 48), fill=fade_color(col_end, 0.7, (0, 0, 0)))
        if prog_real >= 0.99:
            d.text((rx_e + 24, ry_e - 10), "ACTUAL MEASURED", font=get_font("mono_r", 14), fill=WHITE)

    # Delta annotation Δ +80 °C cleanly centered inside lower red wedge!
    if show_delta > 0.05:
        mx = pred_pts[-1][0] + (real_pts[-1][0] - pred_pts[-1][0]) * 0.24
        my = oy + (10.3 / max_depth_km) * gh
        d.text((mx, my), "Δ +80 °C", font=get_font("mono_b", 32), fill=fade_color(RED, show_delta))
        gd.text((mx, my), "Δ +80 °C", font=get_font("mono_b", 32), fill=fade_color(RED, show_delta * 0.6, (0, 0, 0)))
        # Horizontal span indicator between 100°C and 180°C at 12 km
        y12 = pred_pts[-1][1]
        d.line([(pred_pts[-1][0], y12), (real_pts[-1][0], y12)], fill=fade_color(RED, show_delta * 0.7), width=1)

    # Optional Shot 21a dashed red extrapolation 12 km (180°C) -> 15 km (≈ 300°C)!
    if prog_extrap > 0.01 and max_depth_km >= 15.0:
        n_ex = 40
        ex_pts = []
        for i in range(n_ex + 1):
            u = i / n_ex
            dk = 12.0 + 3.0 * u
            tc = 180.0 + 120.0 * (u ** 1.15)
            ex_pts.append((ox + (tc / max_temp_c) * gw, oy + (dk / max_depth_km) * gh))
        ke = int(round(clamp01(prog_extrap) * n_ex))
        if ke >= 2:
            draw_dashed_polyline(d, ex_pts[:ke + 1], RED, width=3, dash=12, gap=8)
            draw_dashed_polyline(gd, ex_pts[:ke + 1], RED, width=3, dash=12, gap=8)
            ex_x, ex_y = ex_pts[ke]
            d.ellipse([ex_x - 7, ex_y - 7, ex_x + 7, ex_y + 7], fill=RED)
            gd.ellipse([ex_x - 9, ex_y - 9, ex_x + 9, ex_y + 9], fill=RED)
            tc_now = int(round(180 + 120 * (ke / n_ex) ** 1.15))
            d.text((ex_x - 140, ex_y - 112), f"≈ {tc_now} °C", font=get_font("mono_b", 44), fill=RED)
            gd.text((ex_x - 140, ex_y - 112), f"≈ {tc_now} °C", font=get_font("mono_b", 44), fill=(190, 45, 30))
            if prog_extrap > 0.9:
                d.text((ex_x - 175, ex_y - 56), "PROJECTED AT 15 km · DRILL LIMIT", font=get_font("mono_r", 14), fill=WHITE)

    return d

# ---------- SHOT 11 (01:46 - 02:06 | MG Signature Temperature Divergence 100°C vs 180°C) ----------
def render_shot11(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.25)

    abs_t = 106.0 + t
    t_lock = ctx["cues"]["shot11_180_lock"] - 106.0  # ~8.2s local time!

    prog_pred = ease_out_cubic(clamp01((t - 0.3) / 3.8))
    prog_real = clamp01((t - 4.4) / max(1.8, t_lock - 4.4))
    show_delta = ease_out_cubic(clamp01((t - (t_lock - 0.4)) / 1.2))

    # Left HUD depth ruler descends synchronously with active curve!
    depth_m = 12000.0 * max(prog_pred, prog_real)
    cur_col = lerp_color(AMBER, RED, clamp01((prog_real - 0.4) / 0.6)) if prog_real > 0.05 else AMBER
    draw_hud_depth_ruler(d, gd, depth_m, 12000.0, alpha=1.0, highlight_color=cur_col)

    _draw_temp_profile(im, glow, d, gd, prog_pred, prog_real, 0.0, 12.0, 200.0, show_delta)
    return add_bloom(im, glow, radius=6, strength=0.40)

# ---------- SHOT 12 (02:06 - 02:21 | Hybrid Viscous Plastic-Flowing Heated Granite Macro) ----------
def render_shot12(t: float, dur: float, ctx: dict) -> Image.Image:
    prog = clamp01(t / dur)
    # True continuous viscous fluid-shear warp field (computed at half-res for smooth DOF optics & 12x speedup)
    if "shot12_half" not in ctx:
        base_h = ctx["plates"]["shot12_granite"].resize(
            (ctx["plates"]["shot12_granite"].width // 2, ctx["plates"]["shot12_granite"].height // 2),
            Image.Resampling.BILINEAR,
        )
        arr_h = np.asarray(base_h, dtype=np.uint8)
        ph, pw, _ = arr_h.shape
        yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
        ctx["shot12_half"] = (arr_h, xx + (pw - W // 2) * 0.5, yy + (ph - H // 2) * 0.5)
    arr, gx, gy = ctx["shot12_half"]
    ph, pw, _ = arr.shape

    # Smooth non-linear creeping viscous flow growing slowly over 15 seconds
    amp = 21.0 * (prog ** 1.15)
    dx = amp * np.sin(gy / 87.5 + t * 0.22) + (amp * 0.45) * np.cos((gx + gy) / 130.0 - t * 0.14)
    dy = amp * 0.65 * np.cos(gx / 105.0 - t * 0.18) + (amp * 0.35) * np.sin(gy / 60.0 + t * 0.16)

    sx = np.clip(gx + dx, 0, pw - 2)
    sy = np.clip(gy + dy, 0, ph - 2)
    ix = sx.astype(np.int32)
    iy = sy.astype(np.int32)
    fx = (sx - ix)[:, :, None]
    fy = (sy - iy)[:, :, None]
    warped = (arr[iy, ix] * (1 - fx) * (1 - fy) + arr[iy, ix + 1] * fx * (1 - fy) +
              arr[iy + 1, ix] * (1 - fx) * fy + arr[iy + 1, ix + 1] * fx * fy)
    warmth = 1.0 + 0.08 * math.sin(t * 0.85)
    warped[:, :, 0] = np.clip(warped[:, :, 0] * warmth, 0, 255)
    im_h = Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))

    # Subtle optical focal-plane drift (microscopic depth-of-field shift)
    blur_r = 0.8 * (0.5 + 0.5 * math.cos(prog * math.pi * 1.4))
    if blur_r > 0.2:
        im_h = im_h.filter(ImageFilter.GaussianBlur(radius=blur_r))
    im = im_h.resize((W, H), Image.Resampling.BILINEAR)

    # Top-right minimalist telemetry HUD
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    al = ease_out_cubic(clamp01((t - 0.6) / 0.8))
    hx, hy = W - 470, 74
    d.rectangle([hx - 18, hy - 14, W - 72, hy + 72], fill=BG, outline=fade_color(AMBER, al * 0.65), width=1)
    d.text((hx, hy), "T 180 °C  ·  P ≈ 3,000 atm", font=get_font("mono_b", 22), fill=fade_color(AMBER, al))
    gd.text((hx, hy), "T 180 °C  ·  P ≈ 3,000 atm", font=get_font("mono_b", 22), fill=fade_color(AMBER, al * 0.5, (0, 0, 0)))
    d.text((hx, hy + 36), "DUCTILE CREEP REGIME · GRANITE", font=get_font("mono_r", 14), fill=fade_color(WHITE, al * 0.72))

    return add_bloom(im, glow, radius=5, strength=0.30)

# ---------- SHOT 13 (02:21 - 02:35 | Hybrid Borehole Wall Creeping Inward & Re-drilling) ----------
def render_shot13(t: float, dur: float, ctx: dict) -> Image.Image:
    # Dark granite rock texture forms left & right borehole walls; central vertical borehole breathes & closes!
    rock = np.asarray(ctx["plates"]["shot13_rockwall"], dtype=np.uint8)
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))

    # Timeline inside 14s shot:
    # 0.0-4.5s: Drill bit withdraws UPWARD (y: 780 -> 140)
    # 3.5-8.5s: Borehole walls below the bit slowly creep INWARD (half-width: 155px -> 78px!)
    # 8.5-11.0s: Drill bit re-descends (y: 140 -> 820), re-cutting walls back open above bit!
    # 11.0-14.0s: Static hold + slight zoom-out feel on constricted lower borehole!
    if t < 4.5:
        bit_y = 780 - 640 * ease_in_out_cubic(t / 4.5)
        creep_amt = ease_in_out_cubic(clamp01((t - 3.2) / 5.0))
        recut = False
    elif t < 8.5:
        bit_y = 140
        creep_amt = ease_in_out_cubic(clamp01((t - 3.2) / 5.0))
        recut = False
    elif t < 11.0:
        p_dn = ease_in_out_cubic((t - 8.5) / 2.5)
        bit_y = 140 + 680 * p_dn
        creep_amt = 1.0
        recut = True
    else:
        bit_y = 820
        creep_amt = 1.0
        recut = True

    # Build rock walls mask cleanly in NumPy so the closing rock looks genuinely organic & massive!
    cx = W // 2
    yy = np.arange(H, dtype=np.float32)
    # Nominal borehole half-width = 155px; inward creep shrinks lower wall organically!
    # Organic vertical wave so rock walls bulge naturally like viscous heated stone
    bulge_profile = np.clip((yy - 200.0) / 480.0, 0.0, 1.0)
    organic = 1.0 + 0.14 * np.sin(yy / 72.0) + 0.08 * np.cos(yy / 31.0)
    inward = 82.0 * creep_amt * bulge_profile * organic
    # If bit is re-descending, clear rock above bit_y!
    if recut:
        cleared = np.clip((bit_y - yy) / 40.0, 0.0, 1.0)
        inward = inward * (1.0 - cleared * 0.82)

    half_w = 155.0 - inward
    # Operate only on active central borehole corridor (x in [cx-220, cx+220]): pixel-identical & 5x faster!
    x0, x1 = cx - 220, cx + 220
    xx = np.arange(x0, x1, dtype=np.float32)[None, :]
    dist_c = np.abs(xx - cx)
    wall_mask = dist_c >= half_w[:, None]

    canvas = rock.copy()
    sub = np.full((H, x1 - x0, 3), BG, dtype=np.uint8)
    rock_sub = rock[:, x0:x1]
    sub[wall_mask] = rock_sub[wall_mask]
    edge_d = np.clip((dist_c - half_w[:, None]) / 26.0, 0.0, 1.0)
    rim_hi = np.exp(-((dist_c - half_w[:, None]) / 6.0) ** 2) * (0.3 + 0.7 * creep_amt)
    cf = sub.astype(np.float32)
    cf[wall_mask] *= (0.55 + 0.45 * edge_d[wall_mask])[:, None]
    cf[:, :, 0] = np.clip(cf[:, :, 0] + rim_hi * 120.0, 0, 255)
    cf[:, :, 1] = np.clip(cf[:, :, 1] + rim_hi * 78.0, 0, 255)
    cf[:, :, 2] = np.clip(cf[:, :, 2] + rim_hi * 14.0, 0, 255)
    canvas[:, x0:x1] = cf.astype(np.uint8)
    im = Image.fromarray(canvas)

    # Subtle pull-back in last 3 seconds (11-14s)
    if t > 11.0:
        p_out = ease_out_cubic((t - 11.0) / 3.0)
        sc = 1.0 - 0.06 * p_out
        nw, nh = int(W * sc), int(H * sc)
        sm = im.resize((nw, nh), Image.Resampling.BILINEAR)
        bg_out = Image.new("RGB", (W, H), BG)
        bg_out.paste(sm, ((W - nw) // 2, (H - nh) // 2))
        im = bg_out

    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    # Original nominal casing dashed reference lines (shows exactly how far rock crept inward!)
    draw_dashed_segment(d, cx - 155, 110, cx - 155, 970, fade_color(WHITE, 0.42), 1, 10, 8)
    draw_dashed_segment(d, cx + 155, 110, cx + 155, 970, fade_color(WHITE, 0.42), 1, 10, 8)

    # Draw steel drill string + heavy rotary tricone drill bit head at bit_y!
    pipe_w = 42
    d.rectangle([cx - pipe_w // 2, 0, cx + pipe_w // 2, bit_y - 36], fill=(58, 64, 72), outline=(145, 152, 160), width=1)
    # Rotating bit flutes
    spin = (t * 28.0) % 14
    for bx in range(cx - 56, cx + 56, 14):
        d.line([(bx + spin * 0.4, bit_y - 36), (bx + spin * 0.4, bit_y + 18)], fill=(115, 122, 130), width=2)
    bit_poly = [(cx - 64, bit_y - 36), (cx + 64, bit_y - 36), (cx + 46, bit_y + 24), (cx, bit_y + 42), (cx - 46, bit_y + 24)]
    d.polygon(bit_poly, fill=(72, 78, 86), outline=AMBER, width=2)
    gd.polygon(bit_poly, outline=AMBER, width=2)

    # Inward lithostatic pressure arrows when creeping!
    if creep_amt > 0.1:
        al_p = clamp01(creep_amt)
        for ay in [420, 560, 700, 840]:
            off = int(18 * math.sin(t * 2.5 + ay))
            # Left arrow pointing right
            lx = cx - 290 + off
            d.line([(lx, ay), (lx + 65, ay)], fill=fade_color(AMBER, al_p * 0.8), width=2)
            d.polygon([(lx + 72, ay), (lx + 60, ay - 6), (lx + 60, ay + 6)], fill=fade_color(AMBER, al_p * 0.8))
            # Right arrow pointing left
            rx = cx + 290 - off
            d.line([(rx, ay), (rx - 65, ay)], fill=fade_color(AMBER, al_p * 0.8), width=2)
            d.polygon([(rx - 72, ay), (rx - 60, ay - 6), (rx - 60, ay + 6)], fill=fade_color(AMBER, al_p * 0.8))

    # Borehole diameter HUD readout
    cur_cm = 23.0 - 9.2 * creep_amt * (0.25 if recut and t > 10.5 else 1.0)
    d.rectangle([W - 430, 72, W - 76, 168], fill=BG, outline=IRON, width=1)
    d.text((W - 408, 86), "BOREHOLE WALL CLOSURE", font=get_font("mono_r", 14), fill=(150, 158, 166))
    d.text((W - 408, 114), f"Ø {cur_cm:4.1f} cm  (NOMINAL 23.0)", font=get_font("mono_b", 22), fill=AMBER)

    draw_hud_depth_ruler(d, gd, 12000.0, 15000.0, alpha=0.85)
    return add_bloom(im, glow, radius=6, strength=0.35)

# ---------- SHOT 14 (02:35 - 02:42 | AI/Archival Core Repository Horizontal Dolly Pan) ----------
def render_shot14(t: float, dur: float, ctx: dict) -> Image.Image:
    pano = ctx["plates"]["shot14_cores_pano"]  # 3320 x 1080
    prog = clamp01(t / dur)
    max_x = pano.width - W
    x0 = int(round(max_x * prog))
    im = pano.crop((x0, 0, x0 + W, H))

    # Subtle warm tungsten pool of light moving across the wooden core boxes (dramatic chiaroscuro!)
    arr = np.asarray(im, dtype=np.float32)
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    lx = W * (0.35 + 0.30 * prog)
    pool = np.exp(-(((xx * 4 - lx) / 620.0) ** 2 + ((yy * 4 - H * 0.48) / 480.0) ** 2))
    pool_f = np.repeat(np.repeat(pool, 4, axis=0), 4, axis=1)
    arr = arr * (0.48 + 0.68 * pool_f[:, :, None])
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    d = ImageDraw.Draw(im)
    d.text((92, 68), "SG-3 CORE ARCHIVE · ZAPOLYARNY, KOLA PENINSULA", font=get_font("mono_r", 15), fill=(185, 175, 160))
    return im

# ---------- SHOT 15 (02:42 - 02:56 | MG Fact 01: No Basalt Layer, Fractured Granite) ----------
def render_shot15(t: float, dur: float, ctx: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.25)

    abs_t = 162.0 + t
    t_nb = ctx["cues"]["shot15_nobasalt_t"] - 162.0  # ~7.25s local time ("没有玄武岩")!

    draw_fact_badge(d, "01", "SEISMIC DISCONTINUITY", ease_out_cubic(clamp01(t / 0.6)))

    # Geological section box
    sx0, sx1 = 360, 1740
    sy0, sy1 = 150, 920
    y_7km = 540  # exact 7 km horizon!
    d.rectangle([sx0, sy0, sx1, y_7km], fill=STRATA[1])
    d.rectangle([sx0, y_7km, sx1, sy1], fill=STRATA[0])
    d.rectangle([sx0, sy0, sx1, sy1], outline=IRON, width=1)

    # Top stratum label
    d.text((sx0 + 28, sy0 + 22), "UPPER GRANITE LAYER  (0 — 7 km)", font=get_font("mono_r", 17), fill=(175, 182, 190))

    # Before t_nb: orderly horizontal dashed strata lines representing assumed basalt transition!
    # After t_nb: strata SHATTER smoothly into jagged high-pressure fracture mesh!
    shatter = ease_out_cubic(clamp01((t - t_nb) / 1.4))
    rng = np.random.default_rng(7001)
    if shatter < 0.99:
        al_str = 1.0 - shatter
        for yy in range(y_7km + 44, sy1, 44):
            draw_dashed_segment(d, sx0 + 16, yy, sx1 - 16, yy, fade_color((75, 88, 102), al_str), 1, 16, 14)
    if shatter > 0.02:
        # Jagged high-pressure metamorphic fracture web across lower half!
        col_f = fade_color((118, 130, 142), shatter * 0.85)
        for _ in range(75):
            fx0 = rng.uniform(sx0 + 20, sx1 - 20)
            fy0 = rng.uniform(y_7km + 10, sy1 - 10)
            ang = rng.uniform(0, 2 * math.pi)
            ln = rng.uniform(55, 175) * shatter
            midx = fx0 + math.cos(ang) * ln * 0.5 + rng.uniform(-16, 16)
            midy = fy0 + math.sin(ang) * ln * 0.5 + rng.uniform(-16, 16)
            fx1 = max(sx0 + 8, min(sx1 - 8, fx0 + math.cos(ang) * ln))
            fy1 = max(y_7km + 6, min(sy1 - 8, fy0 + math.sin(ang) * ln))
            d.line([(fx0, fy0), (midx, midy), (fx1, fy1)], fill=col_f, width=1)

    # 7 km dashed horizon line
    draw_dashed_segment(d, sx0, y_7km, sx1, y_7km, WHITE, width=2, dash=14, gap=8)
    d.text((sx0 - 82, y_7km - 10), "7.0 km", font=get_font("mono_b", 18), fill=AMBER)

    # Solid AMBER borehole drilling straight down through 7 km!
    bx = 760
    drill_p = ease_in_out_cubic(clamp01((t - 1.2) / (t_nb - 0.8)))
    by_tip = sy0 + (sy1 - sy0 - 40) * drill_p
    d.line([(bx, sy0), (bx, by_tip)], fill=AMBER, width=4)
    gd.line([(bx, sy0), (bx, by_tip)], fill=AMBER, width=4)
    d.ellipse([bx - 6, by_tip - 6, bx + 6, by_tip + 6], fill=WHITE)

    # Label right side: "PREDICTED: BASALT" struck through at t_nb, replaced by solid "METAMORPHOSED GRANITE · FRACTURED"
    lx = 920
    strike = ease_out_cubic(clamp01((t - t_nb) / 0.45))
    c_old = lerp_color(WHITE, (92, 98, 106), strike)
    d.text((lx, y_7km + 54), "PREDICTED: BASALT", font=get_font("mono_b", 36), fill=c_old)
    if strike > 0.01:
        d.line([(lx - 8, y_7km + 74), (lx + int(420 * strike), y_7km + 74)], fill=AMBER, width=3)
        gd.line([(lx - 8, y_7km + 74), (lx + int(420 * strike), y_7km + 74)], fill=AMBER, width=3)

    if t >= t_nb + 0.35:
        al_new = ease_out_cubic(clamp01((t - (t_nb + 0.35)) / 0.7))
        d.rectangle([lx - 14, y_7km + 128, lx + 715, y_7km + 225], fill=BG, outline=fade_color(AMBER, al_new), width=2)
        d.text((lx + 8, y_7km + 144), "METAMORPHOSED GRANITE · FRACTURED", font=get_font("mono_b", 28), fill=fade_color(AMBER, al_new))
        gd.text((lx + 8, y_7km + 144), "METAMORPHOSED GRANITE · FRACTURED", font=get_font("mono_b", 28), fill=fade_color(AMBER, al_new * 0.55, (0, 0, 0)))
        d.text((lx + 8, y_7km + 186), "SEISMIC CONRAD DISCONTINUITY = METAMORPHIC TRANSITION", font=get_font("mono_r", 14), fill=fade_color(WHITE, al_new * 0.75))

    return add_bloom(im, glow, radius=6, strength=0.35)

# ---------- SHOT 16 (02:56 - 03:09 | Hybrid Fact 02: 3D Crystal Lattice Squeezing H2O -> Wet Rock Fissure) ----------
def _render_lattice_frame(t: float) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)
    draw_mg_grid(d, 0.28)

    # True 3D perspective rotating cubic/mineral crystal lattice (3x3x3 nodes)
    ang_y = t * 0.55 + 0.4
    ang_x = 0.38 + 0.12 * math.sin(t * 0.6)
    squeeze = ease_in_out_cubic(clamp01((t - 1.5) / 3.8))
    cy_s, sy_s = math.cos(ang_y), math.sin(ang_y)
    cx_s, sx_s = math.cos(ang_x), math.sin(ang_x)

    def proj(x, y, z):
        # Vertical compression under lithostatic pressure!
        y *= (1.0 - 0.26 * squeeze)
        x *= (1.0 + 0.08 * squeeze)
        # Rotate Y then X
        x1 = x * cy_s + z * sy_s
        z1 = -x * sy_s + z * cy_s
        y2 = y * cx_s - z1 * sx_s
        z2 = y * sx_s + z1 * cx_s
        f = 760.0 / (z2 + 5.4)
        return (W * 0.52 + x1 * f, H * 0.52 + y2 * f, z2)

    pts = {}
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                pts[(ix, iy, iz)] = proj(ix * 1.15, iy * 1.15, iz * 1.15)

    # Draw wireframe crystal lattice edges sorted by depth
    edges = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    nb = (ix + dx, iy + dy, iz + dz)
                    if nb in pts:
                        p1, p2 = pts[(ix, iy, iz)], pts[nb]
                        edges.append(((p1[2] + p2[2]) * 0.5, p1, p2))
    edges.sort(key=lambda e: -e[0])
    for zmid, p1, p2 in edges:
        dep = clamp01(1.0 - (zmid + 2.2) / 4.4)
        col = fade_color(WHITE, 0.25 + 0.55 * dep)
        d.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=col, width=1)
    for k, (px, py, pz) in pts.items():
        dep = clamp01(1.0 - (pz + 2.2) / 4.4)
        r = 3 + 2 * dep
        d.ellipse([px - r, py - r, px + r, py + r], fill=fade_color(WHITE, 0.4 + 0.6 * dep))

    # Glowing AMBER H2O water molecule nodes squeezed OUTWARD from interstitial crystal sites!
    rng = np.random.default_rng(1602)
    for m_idx in range(16):
        bx = rng.uniform(-0.85, 0.85)
        by = rng.uniform(-0.75, 0.75)
        bz = rng.uniform(-0.85, 0.85)
        # Expulsion vector pushes H2O out horizontally as vertical pressure squeezes crystal
        dir_x = 1.0 if bx >= 0 else -1.0
        dir_z = 1.0 if bz >= 0 else -1.0
        exp_t = ease_out_cubic(clamp01((t - (1.8 + (m_idx % 5) * 0.25)) / 3.0))
        mx = bx + dir_x * 1.45 * exp_t
        my = by * (1.0 - 0.3 * exp_t)
        mz = bz + dir_z * 1.15 * exp_t
        px, py, pz = proj(mx, my, mz)
        r = 5 + 2 * clamp01(1.0 - (pz + 2.2) / 4.4)
        d.ellipse([px - r, py - r, px + r, py + r], fill=AMBER)
        gd.ellipse([px - r - 2, py - r - 2, px + r + 2, py + r + 2], fill=AMBER)
        if m_idx == 0 and exp_t > 0.35:
            d.text((px + 14, py - 10), "H₂O", font=get_font("mono_b", 18), fill=AMBER)

    # Top & bottom lithostatic pressure arrows pressing inward onto lattice!
    al_arr = ease_out_cubic(clamp01((t - 1.0) / 0.8))
    push = int(38 * squeeze)
    for ax in [W * 0.52 - 140, W * 0.52, W * 0.52 + 140]:
        # Top pointing down
        y0 = 135 + push
        d.line([(ax, y0), (ax, y0 + 55)], fill=fade_color(WHITE, al_arr * 0.75), width=2)
        d.polygon([(ax, y0 + 65), (ax - 6, y0 + 53), (ax + 6, y0 + 53)], fill=fade_color(WHITE, al_arr * 0.75))
        # Bottom pointing up
        yb = 945 - push
        d.line([(ax, yb), (ax, yb - 55)], fill=fade_color(WHITE, al_arr * 0.75), width=2)
        d.polygon([(ax, yb - 65), (ax - 6, yb - 53), (ax + 6, yb - 53)], fill=fade_color(WHITE, al_arr * 0.75))

    d.text((W - 470, H - 105), "HYDROXYL DEHYDRATION UNDER PRESSURE", font=get_font("mono_r", 15), fill=(160, 168, 176))
    return add_bloom(im, glow, radius=6, strength=0.45)

def _render_fissure_frame(tl: float) -> Image.Image:
    return None  # rendered directly inside render_shot16 with ctx plates

def render_shot16(t: float, dur: float, ctx: dict) -> Image.Image:
    # 0 - 6.0s: 3D mineral crystal lattice; 6.0 - 7.0s: smooth cross-dissolve; 7.0 - 13.0s: wet deep rock fissure with animated sliding water beads!
    if t < 6.0:
        im = _render_lattice_frame(t)
    else:
        tl = t - 6.0
        fiss = ken_burns(ctx["plates"]["shot16b_fissure"], clamp01(tl / 7.0), z0=1.00, z1=1.07, cx0=0.50, cy0=0.50, cx1=0.52, cy1=0.52)
        # Overlay animated glistening liquid water droplets beading & sliding along the rock fissure!
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        for k in range(18):
            # Follow the sculpted fissure curve across the frame
            u = ((k / 18.0) + tl * (0.022 + 0.008 * (k % 3))) % 1.0
            dx = u * W
            dy = H * 0.44 + (dx - W * 0.5) * 0.12 + 34.0 * math.sin(dx / 145.0) + 12.0 * math.cos(dx / 45.0)
            # Some droplets drip slightly downward from the lower lip of the fissure!
            drip = ((tl * 0.35 + k * 0.41) % 1.0)
            drop_y = dy + (drip ** 2.4) * (28.0 if k % 2 == 0 else 6.0)
            rad = 3.0 + 2.5 * ((k % 3) / 2.0)
            od.ellipse([dx - rad, drop_y - rad * 1.2, dx + rad, drop_y + rad * 1.2], fill=(185, 208, 224, 145), outline=(235, 245, 255, 210), width=1)
            # Tiny specular glint dot inside water droplet
            od.ellipse([dx - rad * 0.4, drop_y - rad * 0.6, dx - rad * 0.4 + 2, drop_y - rad * 0.6 + 2], fill=(255, 255, 255, 235))
        # Subtle label indicating impermeable cap rock above
        od.text((W - 430, 82), "IMPERMEABLE CAP ROCK · SEALED", font=get_font("mono_r", 15), fill=(190, 198, 206, 190))
        od.text((W - 430, 108), "DEEP FREE WATER · 4—9 km", font=get_font("mono_b", 18), fill=(*AMBER, 220))
        fiss = Image.alpha_composite(fiss.convert("RGBA"), ov).convert("RGB")

        if t < 7.0:
            lat = _render_lattice_frame(t)
            alpha = ease_in_out_cubic(t - 6.0)
            im = Image.blend(lat, fiss, alpha)
        else:
            im = fiss

    d = ImageDraw.Draw(im)
    draw_fact_badge(d, "02", "DEEP FREE WATER", ease_out_cubic(clamp01(t / 0.6)))
    return im

# ---------- SHOT 17 (03:09 - 03:19 | AI/Archival Fact 03: Boiling Hydrogen Mud Pit) ----------
def render_shot17(t: float, dur: float, ctx: dict) -> Image.Image:
    # Subtle micro-push on real dark-graded bubbling mud pit
    im = ken_burns(ctx["plates"]["shot17_mud"], clamp01(t / dur), z0=1.02, z1=1.06, cx0=0.50, cy0=0.50, cx1=0.50, cy1=0.51)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)

    # Procedural concentric mud-bubble burst ripples inside the crater center!
    bubbles = [
        (980, 560, 0.0, 1.8), (890, 610, 0.6, 1.6), (1055, 595, 1.1, 2.0),
        (935, 515, 0.3, 1.5), (1010, 645, 0.9, 1.9), (860, 555, 1.4, 1.7),
    ]
    for bx, by, phase, period in bubbles:
        p = ((t + phase) % period) / period
        r = 10 + p * 92
        al = int((1.0 - p) * 185)
        od.ellipse([bx - r, by - r * 0.62, bx + r, by + r * 0.62], outline=(222, 228, 234, al), width=2)
        if p < 0.25:
            br = 8 + (p / 0.25) * 16
            od.ellipse([bx - br, by - br * 0.75, bx + br, by + br * 0.75], fill=(108, 114, 122, 130), outline=(235, 240, 244, 210), width=1)

    # Rising wispy cold Arctic white vapor plumes (quarter-res blur -> 16x faster & visually identical soft fog)
    rng = np.random.default_rng(1703)
    vapor_q = Image.new("RGBA", (W // 4, H // 4), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vapor_q)
    for k in range(28):
        x0 = rng.uniform(620, 1320)
        y_start = rng.uniform(520, 740)
        speed = rng.uniform(55, 125)
        life = ((t * 0.24 + k * 0.19) % 1.0)
        px = (x0 + 48.0 * math.sin(t * 0.8 + k) + life * 40.0) * 0.25
        py = (y_start - life * speed * 4.2) * 0.25
        rad = (55 + life * 95) * 0.25
        al = int(math.sin(life * math.pi) * 28)
        vd.ellipse([px - rad, py - rad * 0.7, px + rad, py + rad * 0.7], fill=(217, 221, 224, al))

    vapor = vapor_q.filter(ImageFilter.GaussianBlur(radius=4.5)).resize((W, H), Image.Resampling.BILINEAR)
    im = Image.alpha_composite(im.convert("RGBA"), ov)
    im = Image.alpha_composite(im, vapor).convert("RGB")
    # Soft subtle specular amber warning-beacon reflection modulated by wet mud luminance (never a flat oval!)
    arr = np.asarray(im, dtype=np.float32)
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    pulse_b = 0.55 + 0.45 * math.sin(t * 2.6)
    refl = np.exp(-(((xx * 4 - 965.0) / 300.0) ** 2 + ((yy * 4 - 570.0) / 210.0) ** 2))
    refl_f = np.repeat(np.repeat(refl, 4, axis=0), 4, axis=1)
    lum = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) / (3.0 * 255.0)
    shimmer = refl_f * np.clip((lum - 0.18) * 2.5, 0.0, 1.0) * 62.0 * pulse_b
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shimmer * 1.0, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + shimmer * 0.65, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shimmer * 0.05, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8))
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    gd = ImageDraw.Draw(glow)

    draw_fact_badge(d, "03", "HYDROGEN-SATURATED MUD", ease_out_cubic(clamp01(t / 0.6)))

    # Crisp thin frost-white HUD chemical formula H2 bottom right!
    al_h2 = ease_out_cubic(clamp01((t - 3.6) / 0.9))
    if al_h2 > 0.01:
        hx, hy = W - 265, H - 230
        d.text((hx, hy), "H", font=get_font("mono_r", 92), fill=fade_color(WHITE, al_h2))
        d.text((hx + 58, hy + 48), "2", font=get_font("mono_r", 52), fill=fade_color(AMBER, al_h2))
        gd.text((hx + 58, hy + 48), "2", font=get_font("mono_r", 52), fill=fade_color(AMBER, al_h2 * 0.65, (0, 0, 0)))
        d.text((hx - 14, hy + 114), "MOLECULAR HYDROGEN", font=get_font("mono_r", 13), fill=fade_color((165, 172, 180), al_h2))

    return add_bloom(im, glow, radius=6, strength=0.35)

# ---------- SHOT 18 (03:19 - 03:35 | Hybrid Fact 04: Dark-Field Microscopy 2.0 Ga Microfossil) ----------
def _build_microfossil_plate() -> Image.Image:
    """Synthesize a jaw-dropping dark-field optical microscopy specimen of a 2.0 Ga spherical
    organic microfossil (acritarch) on pure #0A0A0C with delicate amber rim-light refraction."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W * 0.50, H * 0.47
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)

    # Organic slightly undulating cell wall radius (~248 px)
    rad0 = (248.0
            + 9.5 * np.sin(3 * ang + 0.5)
            + 6.2 * np.cos(5 * ang - 1.1)
            + 3.5 * np.sin(9 * ang + 2.2)
            + 1.8 * np.cos(17 * ang))

    # Multi-scale organic surface texture inside fossil (cellular wall folds + carbon/nitrogen granules)
    tex = (np.sin(dx / 24.0 + np.cos(dy / 29.0) * 1.4) * 0.35 +
           np.cos(dy / 19.0 - np.sin(dx / 21.0) * 1.2) * 0.35 +
           np.sin((dx + dy) / 11.0) * 0.18 +
           np.cos((dx - dy) / 7.5) * 0.12)
    # Fine microscopic organic grain
    rng = np.random.default_rng(2000000)
    micro_grain = rng.normal(0, 0.14, (H, W)).astype(np.float32)

    inside = r < rad0
    # Outer dark-field double membrane + sharp glowing refractive wall rim!
    wall_outer = np.exp(-((r - rad0) / 5.5) ** 2)
    wall_inner = np.exp(-((r - (rad0 - 15.0)) / 4.2) ** 2) * 0.65
    # Directional warm amber side-lighting from upper-right (angle ~ -0.65 rad)
    side_light = 0.42 + 0.58 * np.clip(np.cos(ang + 0.65), -0.35, 1.0)

    # Organic interior folding ridges (dark-field translucent chitin/kerogen membrane)
    fold1 = np.exp(-(((dx * 0.72 + dy * 0.69 - 35.0 * np.sin(dy / 90.0)) / 24.0) ** 2)) * inside
    fold2 = np.exp(-(((dx * -0.45 + dy * 0.89 + 55.0 + 25.0 * np.cos(dx / 85.0)) / 18.0) ** 2)) * inside
    inner_body = inside.astype(np.float32) * np.clip(0.16 + 0.18 * (r / rad0) ** 2 + 0.14 * tex + 0.08 * micro_grain, 0.02, 0.75)

    # Microscopic outer wall spines / ornamentation (acritarch processes)
    spines = (np.maximum(0.0, np.cos(34.0 * ang + 0.3 * np.sin(7 * ang))) ** 16) * np.exp(-((r - (rad0 + 12.0)) / 9.5) ** 2) * 0.55

    # Faint dark-field optical diffraction halo around specimen
    halo = np.exp(-((r - rad0) / 38.0) ** 2) * 0.16 * side_light

    # Assemble rich amber-gold & translucent ivory dark-field microscopy palette on #0A0A0C!
    lum_warm = (wall_outer * 1.15 + wall_inner * 0.75 + spines * 0.8 + fold1 * 0.32 + fold2 * 0.24) * side_light + halo
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    canvas[:, :, 0] = 10/255.0 + inner_body * 0.45 + lum_warm * 1.00
    canvas[:, :, 1] = 10/255.0 + inner_body * 0.34 + lum_warm * 0.68
    canvas[:, :, 2] = 12/255.0 + inner_body * 0.24 + lum_warm * 0.22

    # Tiny floating dark-field dust motes far in background (out of focus)
    for _ in range(45):
        mx, my = rng.uniform(140, W - 140), rng.uniform(120, H - 120)
        if math.hypot(mx - cx, my - cy) > 310:
            mr = rng.uniform(2.0, 5.5)
            m_al = rng.uniform(0.06, 0.18)
            d2 = (xx - mx) ** 2 + (yy - my) ** 2
            mote = np.exp(-d2 / (mr * mr)) * m_al
            canvas[:, :, 0] += mote * 0.85
            canvas[:, :, 1] += mote * 0.78
            canvas[:, :, 2] += mote * 0.65

    return Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="RGB")

def render_shot18(t: float, dur: float, ctx: dict) -> Image.Image:
    if "microfossil_plate" not in ctx:
        ctx["microfossil_plate"] = _build_microfossil_plate()
    plate = ctx["microfossil_plate"]

    # Slow optical microscope push-in from 0 -> 14s, then DEAD STILLNESS for last 2s!
    move_p = ease_out_cubic(clamp01(t / 13.8))
    im = ken_burns(plate, move_p, z0=1.00, z1=1.10, cx0=0.50, cy0=0.50, cx1=0.50, cy1=0.49)

    # Optical rack-focus pull: begins softly defocused (radius 8px) -> locks razor sharp at t = 4.2s!
    if t < 4.2:
        blur_r = 8.0 * ((1.0 - ease_out_cubic(t / 4.2)) ** 1.3)
        if blur_r > 0.3:
            im = im.filter(ImageFilter.GaussianBlur(radius=blur_r))

    d = ImageDraw.Draw(im)
    draw_fact_badge(d, "04", "PALEOPROTEROZOIC MICROFOSSILS", ease_out_cubic(clamp01(t / 0.6)))

    # Ultra-delicate microscope optical crosshair reticle ticks around edges
    cx, cy = W // 2, int(H * 0.47)
    c_ret = (65, 70, 76)
    d.line([(cx - 340, cy), (cx - 315, cy)], fill=c_ret, width=1)
    d.line([(cx + 315, cy), (cx + 340, cy)], fill=c_ret, width=1)
    d.line([(cx, cy - 320), (cx, cy - 295)], fill=c_ret, width=1)
    d.line([(cx, cy + 295), (cx, cy + 320)], fill=c_ret, width=1)

    # Bottom microscopy telemetry HUD + clean schematic scale bar
    al_h = ease_out_cubic(clamp01((t - 1.5) / 1.0))
    if al_h > 0.01:
        d.text((104, H - 154), "24 SPECIES  ·  ~2.0 Ga  ·  DEPTH > 6,000 m", font=get_font("mono_b", 22), fill=fade_color(WHITE, al_h * 0.88))
        d.text((104, H - 122), "ORGANIC-WALLED PLANKTON · CARBON / NITROGEN MATRIX", font=get_font("mono_r", 14), fill=fade_color((140, 148, 156), al_h))
        # Schematic scale bar bottom right
        sb_x1 = W - 104
        sb_x0 = sb_x1 - 150
        sb_y = H - 134
        c_sb = fade_color(WHITE, al_h * 0.82)
        d.line([(sb_x0, sb_y), (sb_x1, sb_y)], fill=c_sb, width=2)
        d.line([(sb_x0, sb_y - 6), (sb_x0, sb_y + 6)], fill=c_sb, width=2)
        d.line([(sb_x1, sb_y - 6), (sb_x1, sb_y + 6)], fill=c_sb, width=2)
        d.text((sb_x0 + 18, sb_y - 26), "SCALE (SCHEMATIC)", font=get_font("mono_r", 12), fill=fade_color((145, 152, 160), al_h))

    return im
