#!/usr/bin/env python3
"""
Shared visual primitives, typography, HUD components, bloom, film grain,
and Ken Burns / warp helpers matching Part 1 Visual Bible.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 24

# Part 1.1 Exact Hex Palette
BG = (10, 10, 12)        # #0A0A0C Abyssal Black
IRON = (58, 63, 68)      # #3A3F44 Iron Grey
WHITE = (217, 221, 224)  # #D9DDE0 Permafrost White
BLUE = (22, 34, 46)      # #16222E Polar-Night Blue
GRID = (36, 39, 43)      # #24272B Grid Grey
AMBER = (255, 176, 0)    # #FFB000 Warning Amber
RED = (229, 64, 42)      # #E5402A Alert Red (strictly rationed!)
RUST = (122, 52, 38)     # #7A3426 Rust Red
STRATA = [(22, 34, 46), (30, 42, 53), (38, 50, 61)]

# Fonts (verified SC indices!)
FONT_SERIF_B = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
FONT_SANS_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_SANS_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_MONO_R = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

_FONT_CACHE = {}

def get_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    key = (kind, size)
    if key not in _FONT_CACHE:
        if kind == "serif_b":
            _FONT_CACHE[key] = ImageFont.truetype(FONT_SERIF_B, size, index=2)
        elif kind == "sans_r":
            _FONT_CACHE[key] = ImageFont.truetype(FONT_SANS_R, size, index=2)
        elif kind == "sans_b":
            _FONT_CACHE[key] = ImageFont.truetype(FONT_SANS_B, size, index=2)
        elif kind == "mono_r":
            _FONT_CACHE[key] = ImageFont.truetype(FONT_MONO_R, size)
        elif kind == "mono_b":
            _FONT_CACHE[key] = ImageFont.truetype(FONT_MONO_B, size)
        else:
            raise ValueError(kind)
    return _FONT_CACHE[key]

def ease_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3

def ease_in_out_cubic(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 4 * x * x * x if x < 0.5 else 1.0 - ((-2 * x + 2) ** 3) / 2.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def lerp_color(c1, c2, t: float):
    t = clamp01(t)
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))

def fade_color(c, alpha: float, bg=BG):
    return lerp_color(bg, c, clamp01(alpha))

# Precomputed 24-frame tileable 35mm organic film grain bank (fast uint8 additive)
_GRAIN_BANK = None

def init_grain_bank():
    global _GRAIN_BANK
    if _GRAIN_BANK is not None:
        return
    rng = np.random.default_rng(198912)
    # 24 frames of 270x480 noise upscaled 4x? Better: native 540x960 tiled 2x2 for authentic 35mm fine grain!
    bank = []
    for _ in range(24):
        g = rng.normal(0.0, 1.0, (H // 2, W // 2)).astype(np.float32)
        # Repeat 2x2 for realistic 35mm grain size at 1080p, super fast!
        g2 = np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)
        bank.append(g2)
    _GRAIN_BANK = bank

def apply_film_grain(im: Image.Image, frame_idx: int, strength: float = 6.5) -> np.ndarray:
    """Returns uint8 RGB numpy array (H, W, 3) with authentic 35mm grain."""
    init_grain_bank()
    arr = np.asarray(im, dtype=np.int16)
    g = (_GRAIN_BANK[frame_idx % 24] * strength).astype(np.int16)
    arr = arr + g[:, :, None]
    return np.clip(arr, 0, 255).astype(np.uint8)

def draw_tracked_text(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str,
                      font: ImageFont.FreeTypeFont, fill, tracking_px: float = 0.0,
                      anchor: str = "mm"):
    """Draw text with exact custom letter spacing (tracking), centered or left-aligned."""
    if not text:
        return
    widths = [draw.textlength(ch, font=font) for ch in text]
    total_w = sum(widths) + tracking_px * max(0, len(text) - 1)
    bbox = font.getbbox(text)
    text_h = bbox[3] - bbox[1]
    if anchor[0] == "m":
        x = cx - total_w / 2.0
    elif anchor[0] == "r":
        x = cx - total_w
    else:
        x = cx
    if anchor[1] == "m":
        y = cy - text_h / 2.0 - bbox[1]
    elif anchor[1] == "b":
        y = cy - text_h - bbox[1]
    else:
        y = cy - bbox[1]
    for ch, cw in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += cw + tracking_px

def draw_dashed_segment(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, fill, width=2, dash=12, gap=8):
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(dist, pos + dash)
        draw.line([(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)], fill=fill, width=width)
        pos += dash + gap

def draw_dashed_polyline(draw: ImageDraw.ImageDraw, pts, fill, width=2, dash=12, gap=8):
    if len(pts) < 2:
        return
    carry = 0.0
    drawing = True
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-4:
            continue
        ux, uy = dx / seg, dy / seg
        pos = 0.0
        while pos < seg:
            rem = (dash if drawing else gap) - carry
            step = min(rem, seg - pos)
            if drawing:
                draw.line([(x0 + ux * pos, y0 + uy * pos),
                           (x0 + ux * (pos + step), y0 + uy * (pos + step))], fill=fill, width=width)
            pos += step
            carry += step
            if carry >= (dash if drawing else gap) - 1e-4:
                carry = 0.0
                drawing = not drawing

def draw_dashed_rect(draw: ImageDraw.ImageDraw, box, fill, width=1, dash=10, gap=7):
    x0, y0, x1, y1 = box
    draw_dashed_segment(draw, x0, y0, x1, y0, fill, width, dash, gap)
    draw_dashed_segment(draw, x1, y0, x1, y1, fill, width, dash, gap)
    draw_dashed_segment(draw, x1, y1, x0, y1, fill, width, dash, gap)
    draw_dashed_segment(draw, x0, y1, x0, y0, fill, width, dash, gap)

def add_bloom(base_im: Image.Image, glow_im: Image.Image, radius: float = 6.0, strength: float = 0.35) -> Image.Image:
    """Fast optical Gaussian bloom composited additively onto base_im (half-res blur -> 4x faster)."""
    small = glow_im.resize((W // 2, H // 2), Image.Resampling.BILINEAR)
    blurred = small.filter(ImageFilter.GaussianBlur(radius=max(1.0, radius * 0.5))).resize(
        (W, H), Image.Resampling.BILINEAR
    )
    b_arr = np.asarray(base_im, dtype=np.int16)
    g_arr = np.asarray(glow_im, dtype=np.int16)
    bl_arr = (np.asarray(blurred, dtype=np.float32) * strength).astype(np.int16)
    out = np.clip(b_arr + g_arr + bl_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def ken_burns(plate: Image.Image, prog: float, z0: float = 1.0, z1: float = 1.08,
              cx0: float = 0.5, cy0: float = 0.5, cx1: float = 0.5, cy1: float = 0.5) -> Image.Image:
    """Sub-pixel smooth zoom/pan crop of an overscanned plate to exact 1920x1080."""
    pw, ph = plate.size
    z = z0 + (z1 - z0) * prog
    cx = cx0 + (cx1 - cx0) * prog
    cy = cy0 + (cy1 - cy0) * prog
    # Crop box size keeping 16:9
    cw = pw / z
    ch = cw * H / W
    if ch > ph:
        ch = ph / z
        cw = ch * W / H
    x0 = cx * pw - cw / 2.0
    y0 = cy * ph - ch / 2.0
    x0 = max(0.0, min(pw - cw, x0))
    y0 = max(0.0, min(ph - ch, y0))
    return plate.transform(
        (W, H),
        Image.Transform.EXTENT,
        (x0, y0, x0 + cw, y0 + ch),
        resample=Image.Resampling.BILINEAR,
    )

def draw_hud_depth_ruler(draw: ImageDraw.ImageDraw, glow_draw: ImageDraw.ImageDraw,
                         current_depth_m: float, max_depth_m: float = 15000.0,
                         alpha: float = 1.0, highlight_color=AMBER):
    """Part 1.2 persistent left-margin HUD Depth Ruler (fixed at x = 8% = 154px)."""
    if alpha <= 0.01:
        return
    rx = 154
    y_top, y_bot = 140, 940
    span = y_bot - y_top
    c_line = fade_color((80, 86, 94), alpha)
    c_tick = fade_color((110, 116, 124), alpha)
    c_txt = fade_color((145, 152, 160), alpha)
    c_hi = fade_color(highlight_color, alpha)

    # Top permanent coordinate stamp
    f_small = get_font("mono_r", 15)
    f_val = get_font("mono_b", 18)
    draw.text((76, 62), "69°23′N  30°36′E", font=f_small, fill=c_txt)
    draw.text((76, 84), "KOLA SG-3 · DEPTH", font=get_font("mono_r", 12), fill=fade_color((90, 96, 104), alpha))
    draw.line([(76, 106), (228, 106)], fill=fade_color(GRID, alpha), width=1)

    # Vertical spine
    draw.line([(rx, y_top), (rx, y_bot)], fill=c_line, width=1)

    # Ticks every 200m (minor) and 1,000m (major)
    step_minor = 200
    n_steps = int(max_depth_m // step_minor)
    for i in range(n_steps + 1):
        d_m = i * step_minor
        y = y_top + (d_m / max_depth_m) * span
        if d_m % 1000 == 0:
            draw.line([(rx - 14, y), (rx, y)], fill=c_tick, width=1)
            if d_m % 3000 == 0 or d_m == int(max_depth_m):
                lbl = f"{d_m // 1000:2d} km" if d_m > 0 else " 0 km"
                draw.text((rx - 68, y - 8), lbl, font=get_font("mono_r", 13), fill=c_txt)
        else:
            draw.line([(rx - 6, y), (rx, y)], fill=fade_color(GRID, alpha), width=1)

    # Active depth progress bar + amber triangle cursor
    d_clamped = max(0.0, min(max_depth_m, current_depth_m))
    cy = y_top + (d_clamped / max_depth_m) * span
    draw.line([(rx, y_top), (rx, cy)], fill=c_hi, width=2)
    if glow_draw is not None:
        glow_draw.line([(rx, y_top), (rx, cy)], fill=c_hi, width=2)
    tri = [(rx + 4, cy), (rx + 16, cy - 6), (rx + 16, cy + 6)]
    draw.polygon(tri, fill=c_hi)
    if glow_draw is not None:
        glow_draw.polygon(tri, fill=c_hi)
    draw.text((rx + 24, cy - 10), f"-{int(round(d_clamped)):,} m", font=f_val, fill=c_hi)

def draw_mg_grid(draw: ImageDraw.ImageDraw, alpha: float = 0.45):
    """Subtle Part 1.1 #24272B tactical grid background for MG shots."""
    col = fade_color(GRID, alpha)
    for x in range(160, W, 160):
        draw.line([(x, 0), (x, H)], fill=col, width=1)
    for y in range(120, H, 120):
        draw.line([(0, y), (W, y)], fill=col, width=1)

def draw_fact_badge(draw: ImageDraw.ImageDraw, num_str: str, subtitle: str, alpha: float = 1.0):
    """Act 4 crisp monospace corner badge (01 / 02 / 03 / 04)."""
    if alpha <= 0.01:
        return
    c_num = fade_color(AMBER, alpha)
    c_sub = fade_color(WHITE, alpha * 0.72)
    c_bar = fade_color(AMBER, alpha * 0.65)
    draw.line([(86, 68), (86, 134)], fill=c_bar, width=2)
    draw.text((104, 58), num_str, font=get_font("mono_b", 48), fill=c_num)
    draw.text((106, 116), subtitle, font=get_font("mono_r", 14), fill=c_sub)

def draw_snow_streaks(arr: np.ndarray, t: float, count: int = 160, speed_x: float = -1650.0,
                      speed_y: float = 140.0, streak_len: int = 44, intensity: float = 0.42, seed: int = 42):
    """Fast procedural blizzard / drifting snow particles composited cleanly onto RGB uint8/float array."""
    im = Image.fromarray(arr) if isinstance(arr, np.ndarray) else arr
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    rng = np.random.default_rng(seed)
    x0s = rng.uniform(0, W, count)
    y0s = rng.uniform(0, H, count)
    depths = rng.uniform(0.35, 1.35, count)
    for i in range(count):
        dp = depths[i]
        x = (x0s[i] + speed_x * dp * t) % (W + 120) - 60
        y = (y0s[i] + speed_y * dp * t + 18.0 * math.sin(t * 2.1 + i)) % (H + 80) - 40
        al = int(255 * intensity * min(1.0, dp * 0.75))
        if streak_len > 3:
            lx = streak_len * dp * (1.0 if speed_x < 0 else -1.0)
            ly = -streak_len * 0.10 * dp
            d.line([(x, y), (x + lx, y + ly)], fill=(217, 221, 224, al), width=max(1, int(dp * 1.4)))
        else:
            r = max(1, int(dp * 1.8))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(217, 221, 224, al))
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
