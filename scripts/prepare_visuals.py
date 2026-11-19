#!/usr/bin/env python3
"""
Step 3A: Prepare and apply unified Part 1.1 Visual Bible cinema color grading
across all AI-generated plates and curated archival photographs.
"""
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = "/home/antigravity/kola_superdeep"
ART_DIR = "/home/antigravity/.gemini/jetski/brain/v1/default/jetski_remote_agents_3p/8660d99d-b4d4-453e-9341-bb8360253168"
RAW_DIR = f"{ROOT}/assets/raw_photos"
CAND_DIR = f"{ROOT}/assets/candidates"
OUT_DIR = f"{ROOT}/assets/graded"
os.makedirs(OUT_DIR, exist_ok=True)

def find_art(prefix: str) -> str:
    hits = sorted(glob.glob(f"{ART_DIR}/{prefix}*.jpg") + glob.glob(f"{ART_DIR}/{prefix}*.png"))
    assert hits, f"Missing generated image {prefix} in {ART_DIR}"
    return hits[-1]

def cover_crop(im: Image.Image, tw: int, th: int, cx: float = 0.5, cy: float = 0.5) -> Image.Image:
    """Scale and crop to exact (tw, th) preserving aspect ratio, centered at (cx, cy)."""
    w, h = im.size
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x0 = int(round((nw - tw) * cx))
    y0 = int(round((nh - th) * cy))
    x0 = max(0, min(nw - tw, x0))
    y0 = max(0, min(nh - th, y0))
    return im.crop((x0, y0, x0 + tw, y0 + th))

def cinema_grade(
    im: Image.Image,
    sat: float = 0.45,
    contrast: float = 1.24,
    exposure: float = 0.90,
    cool_tint: float = 1.0,
    vignette: float = 0.42,
    kill_green: bool = False,
    keep_rust: bool = False,
    amber_cast: float = 0.0,
) -> Image.Image:
    """Unified Part 1.1 color grading LUT: #0A0A0C shadows, #16222E polar-night blue lows,
    #3A3F44 iron midtones, cool #D9DDE0 highlights, subtle lens vignette."""
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    # Optional green/summer-foliage neutralization (turns Arctic summer grass into winter frost/iron)
    if kill_green:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        greenness = np.clip((g - np.maximum(r, b) * 0.92) * 3.5, 0.0, 1.0)
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        # Desaturate green strongly and cool-frost it slightly
        frost = np.stack([luma * 0.92 + 0.04, luma * 0.96 + 0.04, luma * 1.02 + 0.05], axis=-1)
        arr = arr * (1.0 - greenness[:, :, None] * 0.92) + frost * (greenness[:, :, None] * 0.92)

    # Preserve rust hues selectively if requested (e.g. Shot 23 cap)
    rust_mask = None
    if keep_rust:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rust_mask = np.clip((r - b) * 2.4 - np.clip(g - b, 0, 1) * 0.4, 0.0, 1.0)
        rust_mask = np.clip(rust_mask * 1.4, 0.0, 1.0)

    # Exposure
    arr = np.clip(arr * exposure, 0.0, 1.0)

    # Desaturate toward luminance
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sat_eff = sat
    if keep_rust and rust_mask is not None:
        sat_map = sat + (0.85 - sat) * rust_mask[:, :, None]
        arr = luma[:, :, None] + (arr - luma[:, :, None]) * sat_map
    else:
        arr = luma[:, :, None] + (arr - luma[:, :, None]) * sat_eff

    # Cinematic S-curve contrast around mid-grey (0.45)
    arr = np.clip((arr - 0.45) * contrast + 0.45, 0.0, 1.0)
    # Mild gamma lift/crush so shadows feel dense filmic #0A0A0C
    arr = np.power(arr, 1.08)

    # Split-tone: polar-night blue (#16222E) in shadows, crisp cold permafrost (#D9DDE0) in highlights
    lum2 = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sh_w = np.clip(1.0 - lum2 * 2.1, 0.0, 1.0)[:, :, None] * cool_tint
    hi_w = np.clip((lum2 - 0.55) * 2.2, 0.0, 1.0)[:, :, None] * cool_tint
    # Polar-night blue lift in deep lows
    arr[:, :, 0] = np.clip(arr[:, :, 0] - 0.025 * sh_w[:, :, 0] + 0.005 * hi_w[:, :, 0], 0, 1)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + 0.002 * sh_w[:, :, 0] + 0.010 * hi_w[:, :, 0], 0, 1)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + 0.042 * sh_w[:, :, 0] + 0.018 * hi_w[:, :, 0], 0, 1)

    # Optional warm amber side-glow / tungsten cast
    if amber_cast > 0.0:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + 0.35 * amber_cast), 0, 1)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1.0 + 0.12 * amber_cast), 0, 1)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 - 0.28 * amber_cast), 0, 1)

    # Clamp true black floor to #0A0A0C (10, 10, 12)
    arr[:, :, 0] = arr[:, :, 0] * (1.0 - 10/255.0) + 10/255.0
    arr[:, :, 1] = arr[:, :, 1] * (1.0 - 10/255.0) + 10/255.0
    arr[:, :, 2] = arr[:, :, 2] * (1.0 - 12/255.0) + 12/255.0

    # Smooth cinema lens vignette
    if vignette > 0:
        H, W, _ = arr.shape
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx = (xx - W * 0.5) / (W * 0.55)
        dy = (yy - H * 0.5) / (H * 0.55)
        r2 = np.clip(dx * dx + dy * dy, 0.0, 2.0)
        vig = np.clip(1.0 - vignette * (r2 ** 1.45), 0.18, 1.0)
        base = np.array([10/255.0, 10/255.0, 12/255.0], dtype=np.float32)
        arr = base + (arr - base) * vig[:, :, None]

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")

def build_wet_fissure_plate(granite_path: str) -> Image.Image:
    """Create a dark, moody, photorealistic deep metamorphic granite fracture macro plate
    (heavy impermeable dark rock above, glistening narrow diagonal rock fissure below)."""
    g = Image.open(granite_path).convert("RGB")
    g = cover_crop(g, 2160, 1215)
    # Heavy dark metamorphic rock transformation
    arr = np.asarray(g, dtype=np.float32) / 255.0
    # Invert bright thin-section tones so quartz becomes moody dark grey-blue stone
    arr = 1.0 - arr * 0.72
    H, W, _ = arr.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Create realistic jagged diagonal subterranean fissure across middle-lower rock
    fissure_y = H * 0.44 + (xx - W * 0.5) * 0.12 + 38.0 * np.sin(xx / 150.0) + 14.0 * np.cos(xx / 47.0)
    dist = np.abs(yy - fissure_y)
    # Deep dark crevice + wet glistening specular rock edge
    crevice = np.exp(-((dist / 16.0) ** 2))
    wet_rim = np.exp(-(((dist - 22.0) / 14.0) ** 2)) * 0.55
    # Heavy dense impermeable cap rock pressing down from top 28%
    cap_mask = 1.0 / (1.0 + np.exp((yy - H * 0.25) / 28.0))

    # Apply dark subterranean stone shading
    stone = arr * 0.42
    stone[:, :, 0] *= 0.85
    stone[:, :, 1] *= 0.92
    stone[:, :, 2] *= 1.08
    # Darken crevice to abyssal black, add subtle cool-amber specular highlight along wet edge
    stone = stone * (1.0 - 0.88 * crevice[:, :, None])
    stone[:, :, 0] += wet_rim * 0.14
    stone[:, :, 1] += wet_rim * 0.18
    stone[:, :, 2] += wet_rim * 0.24
    # Heavy oppressive top stratum
    stone = stone * (1.0 - 0.55 * cap_mask[:, :, None])
    im = Image.fromarray((np.clip(stone, 0, 1) * 255).astype(np.uint8))
    return cinema_grade(im, sat=0.35, contrast=1.35, exposure=1.05, cool_tint=1.1, vignette=0.52)

def main():
    saved = []

    def save_plate(name: str, im: Image.Image):
        p = f"{OUT_DIR}/{name}.jpg"
        im.save(p, quality=94)
        saved.append((name, p))
        print(f"Graded {name}: {im.size}")

    # Shot 02: AI CRT TV (already masterfully graded, just subtle unified black-floor & size 2160x1215)
    im = cover_crop(Image.open(find_art("shot02_crt_tv")), 2160, 1215)
    save_plate("shot02_crt", cinema_grade(im, sat=0.88, contrast=1.06, exposure=0.98, cool_tint=0.4, vignette=0.20))

    # Shot 04: AI Ruin Blizzard (exact 1920x1080 static tripod)
    im = cover_crop(Image.open(find_art("shot04_ruin_blizzard")), 1920, 1080)
    save_plate("shot04_blizzard", cinema_grade(im, sat=0.85, contrast=1.08, exposure=0.96, cool_tint=0.5, vignette=0.22))

    # Shot 06a: REAL Kola Soviet tower on snowy Arctic tundra (2160x1215 for slow push-in)
    im = cover_crop(Image.open(f"{RAW_DIR}/06a_kola_tower.jpg"), 2160, 1215, cx=0.5, cy=0.55)
    save_plate("shot06a_tower", cinema_grade(im, sat=0.35, contrast=1.28, exposure=0.86, cool_tint=1.15, vignette=0.40, kill_green=True))

    # Shot 06b: AI Lunar bootprint (1920x1080 dead-still)
    im = cover_crop(Image.open(find_art("shot06b_moon_footprint")), 1920, 1080)
    save_plate("shot06b_moon", cinema_grade(im, sat=0.05, contrast=1.12, exposure=0.95, cool_tint=0.3, vignette=0.25))

    # Shot 08b: REAL Glomar Challenger deep-sea drilling ship (vintage 16mm newsreel)
    im = cover_crop(Image.open(f"{RAW_DIR}/08b_mohole_ship.jpg"), 2160, 1215, cx=0.5, cy=0.45)
    save_plate("shot08b_mohole", cinema_grade(im, sat=0.10, contrast=1.30, exposure=0.85, cool_tint=0.9, vignette=0.50))

    # Shot 09a: REAL Drilling Roughnecks + hydraulic tongs on rig floor (dark moody chiaroscuro)
    im = cover_crop(Image.open(f"{RAW_DIR}/09a_drill_tongs.jpg"), 2160, 1215, cx=0.52, cy=0.58)
    save_plate("shot09a_tongs", cinema_grade(im, sat=0.22, contrast=1.42, exposure=0.68, cool_tint=1.1, vignette=0.58))

    # Shot 12: REAL Alkalic Granite thin-section macro transformed into dark heated deep-earth crystal plate!
    im_g = cover_crop(Image.open(f"{RAW_DIR}/12_granite_macro.jpg"), 2160, 1215)
    arr_g = np.asarray(im_g, dtype=np.float32) / 255.0
    # Darken pale background into deep obsidian-iron, leaving crystal veins lit by warm subterranean amber!
    inv_g = np.power(1.0 - arr_g * 0.85, 1.35)
    H, W, _ = inv_g.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Warm subterranean amber side-lighting sweeping from bottom-right
    glow = np.exp(-(((xx - W * 0.72) / (W * 0.55)) ** 2 + ((yy - H * 0.65) / (H * 0.55)) ** 2))
    lit = np.zeros_like(inv_g)
    lum_g = 0.3 * inv_g[:, :, 0] + 0.59 * inv_g[:, :, 1] + 0.11 * inv_g[:, :, 2]
    # Crisp edge highlight via subtle unsharp contrast
    lit[:, :, 0] = lum_g * (0.45 + 0.95 * glow)
    lit[:, :, 1] = lum_g * (0.42 + 0.58 * glow)
    lit[:, :, 2] = lum_g * (0.50 + 0.10 * glow)
    im_g_dark = Image.fromarray((np.clip(lit, 0, 1) * 255).astype(np.uint8))
    save_plate("shot12_granite", cinema_grade(im_g_dark, sat=0.85, contrast=1.40, exposure=1.15, cool_tint=0.4, vignette=0.48))

    # Shot 13 rock wall texture: dark deep grey metamorphic stone
    im_w = cover_crop(Image.open(f"{RAW_DIR}/12_granite_macro.jpg"), 1920, 1080)
    arr_w = 1.0 - (np.asarray(im_w, dtype=np.float32) / 255.0) * 0.82
    im_w = Image.fromarray((np.clip(arr_w * 0.55, 0, 1) * 255).astype(np.uint8))
    save_plate("shot13_rockwall", cinema_grade(im_w, sat=0.15, contrast=1.35, exposure=0.88, cool_tint=1.0, vignette=0.15))

    # Shot 14: REAL wooden Nordic drill core boxes stitched into wide 3200x1080 horizontal dolly pan!
    c1 = cover_crop(Image.open(f"{RAW_DIR}/14_core_boxes_1.jpg"), 1720, 1080)
    c2 = cover_crop(Image.open(f"{RAW_DIR}/14_core_boxes_2.jpg"), 1720, 1080)
    # Soft 120px crossfade seam so panoramic pan looks seamless
    ov = 120
    pano = Image.new("RGB", (1720 * 2 - ov, 1080))
    a1 = np.asarray(c1, dtype=np.float32)
    a2 = np.asarray(c2, dtype=np.float32)
    out_p = np.zeros((1080, 1720 * 2 - ov, 3), dtype=np.float32)
    out_p[:, :1720 - ov] = a1[:, :1720 - ov]
    alpha = np.linspace(0, 1, ov, dtype=np.float32)[None, :, None]
    out_p[:, 1720 - ov:1720] = a1[:, 1720 - ov:] * (1 - alpha) + a2[:, :ov] * alpha
    out_p[:, 1720:] = a2[:, ov:]
    pano_im = Image.fromarray(out_p.astype(np.uint8))
    save_plate("shot14_cores_pano", cinema_grade(pano_im, sat=0.38, contrast=1.32, exposure=0.72, cool_tint=0.4, vignette=0.15, amber_cast=0.38))

    # Shot 16b: Deep fractured metamorphic granite wet fissure plate
    save_plate("shot16b_fissure", build_wet_fissure_plate(f"{RAW_DIR}/12_granite_macro.jpg"))

    # Shot 17: REAL grey bubbling mud crater dark-graded into industrial mud pit
    im = cover_crop(Image.open(f"{RAW_DIR}/17_bubbling_mud.jpg"), 2160, 1215, cx=0.5, cy=0.52)
    save_plate("shot17_mud", cinema_grade(im, sat=0.18, contrast=1.42, exposure=0.66, cool_tint=1.05, vignette=0.55))

    # Shot 21b: REAL Soviet VVER Cyrillic control room (Kozloduy Unit 5) heavily darkened to Dec 1991 polar night!
    im = cover_crop(Image.open(f"{CAND_DIR}/ctrl_koz.jpg"), 2160, 1215, cx=0.55, cy=0.45)
    save_plate("shot21b_ctrl", cinema_grade(im, sat=0.30, contrast=1.35, exposure=0.52, cool_tint=1.25, vignette=0.55))

    # Shot 22: REAL abandoned Kola Superdeep Borehole facility concrete hall & rusted scrap, winterized!
    im = cover_crop(Image.open(f"{RAW_DIR}/22_kola_ruin_002.jpg"), 1920, 1080, cx=0.5, cy=0.50)
    save_plate("shot22_ruin", cinema_grade(im, sat=0.32, contrast=1.30, exposure=0.78, cool_tint=1.25, vignette=0.46, kill_green=True, keep_rust=True))

    # Shot 23: THE ICONIC ACTUAL KOLA SG-3 BOLTED RUSTED CAP ("12262")!
    # Centered tightly on the bolted metal cap, preserving rich dark rust red (#7A3426)!
    im = cover_crop(Image.open(f"{RAW_DIR}/22_kola_ruin_001.jpg"), 2160, 1215, cx=0.52, cy=0.52)
    save_plate("shot23_cap", cinema_grade(im, sat=0.26, contrast=1.32, exposure=0.80, cool_tint=1.10, vignette=0.52, keep_rust=True))

    # Build graded contact sheet
    cols = 4; cw, ch = 440, 260
    rows = (len(saved) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch), (10, 10, 12))
    draw = ImageDraw.Draw(sheet)
    for i, (name, path) in enumerate(saved):
        r, c = divmod(i, cols)
        im = Image.open(path).convert("RGB")
        im.thumbnail((cw - 12, ch - 30))
        sheet.paste(im, (c * cw + (cw - im.width) // 2, r * ch + 24 + (ch - 30 - im.height) // 2))
        draw.text((c * cw + 8, r * ch + 4), name, fill=(255, 176, 0))
    sheet.save(f"{ROOT}/assets/graded_contact_sheet.jpg", quality=88)
    print(f"Saved graded contact sheet ({len(saved)} plates)!")

if __name__ == "__main__":
    main()
