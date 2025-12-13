import cv2
import numpy as np
import base64
import os

# ---------------------------------------------------
# Output directories
# ---------------------------------------------------
OUTPUT_DIR = "outputs"
OUT_IMAGES = os.path.join(OUTPUT_DIR, "images")
OUT_MASKS = os.path.join(OUTPUT_DIR, "masks")
OUT_SWATCHES = os.path.join(OUTPUT_DIR, "swatches")
OUT_OVERLAYS = os.path.join(OUTPUT_DIR, "overlays")
OUT_PREV = os.path.join(OUTPUT_DIR, "previews")

for d in (OUTPUT_DIR, OUT_IMAGES, OUT_MASKS, OUT_SWATCHES, OUT_OVERLAYS, OUT_PREV):
    os.makedirs(d, exist_ok=True)

def outpath(subdir, fname):
    return os.path.join(subdir, fname)

# ---------------------------------------------------
# CLAHE lighting normalization
# ---------------------------------------------------
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)

    lab2 = cv2.merge((L2, A, B))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------
# Load segmentation PNG (RGBA)
# ---------------------------------------------------
png = cv2.imread("./input_image/person_cutout.png", cv2.IMREAD_UNCHANGED)

if png.shape[2] != 4:
    raise ValueError("Expected RGBA cutout PNG (with alpha).")

bgr = png[:, :, :3]
alpha = png[:, :, 3]

# Normalize lighting
bgr_norm = apply_clahe(bgr)

# ---------------------------------------------------
# Brighten / Saturate adjustment (configurable)
# ---------------------------------------------------
BRIGHTNESS_FACTOR = 1.0 #.15  # multiply V channel in HSV
SATURATION_FACTOR = 1.0 #.25  # multiply S channel in HSV

def adjust_brightness_saturation(img_bgr, brightness=1.0, saturation=1.0):
    """Return a copy of img_bgr with brightness and saturation scaled.

    brightness: multiplier for the V channel (1.0 = no change)
    saturation: multiplier for the S channel (1.0 = no change)
    """
    if brightness == 1.0 and saturation == 1.0:
        return img_bgr.copy()

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    if saturation != 1.0:
        s = s * float(saturation)
    if brightness != 1.0:
        v = v * float(brightness)

    # clip to valid range
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)

    hsv2 = cv2.merge((h, s, v)).astype(np.uint8)
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return bgr2


# Apply brightness/saturation adjustments to the normalized image
bgr_adj = adjust_brightness_saturation(bgr_norm, BRIGHTNESS_FACTOR, SATURATION_FACTOR)
cv2.imwrite(outpath(OUT_IMAGES, "input_adjusted.png"), bgr_adj)

# ---------------------------------------------------
# Auto-detect bounding region from mask
# ---------------------------------------------------
ys, xs = np.where(alpha > 0)
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()
h = y_max - y_min

# Region definitions (same as before)
face_box    = (x_min, x_max, y_min, y_min + int(h*0.20))
hair_box    = (x_min, x_max, y_min, y_min + int(h*0.12))
torso_box   = (x_min, x_max, y_min + int(h*0.20), y_min + int(h*0.55))
legs_box    = (x_min, x_max, y_min + int(h*0.55), y_max)


# ---------------------------------------------------
# Get center pixel of a region
# ---------------------------------------------------
def center_pixel_color(img, mask, box):
    x1, x2, y1, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Move slightly if center lies on transparent area
    step = 3
    tries = 0
    while mask[cy, cx] == 0 and tries < 10:
        cx += step
        cy += step
        tries += 1

    return tuple(map(int, img[cy, cx]))


# ---------------------------------------------------
# Pixel-wise region masks and color stats
# ---------------------------------------------------
def region_mask_from_box(alpha_mask, box):
    """Return a boolean mask (same shape as alpha_mask) for pixels inside
    the provided box and where alpha_mask > 0.
    box = (x1, x2, y1, y2) with x2,y2 exclusive-ish (same semantics as slices).
    """
    x1, x2, y1, y2 = box
    h, w = alpha_mask.shape[:2]
    # clamp box to image
    x1 = max(0, x1); x2 = min(w, x2)
    y1 = max(0, y1); y2 = min(h, y2)

    mask = np.zeros_like(alpha_mask, dtype=bool)
    if y2 > y1 and x2 > x1:
        sub = alpha_mask[y1:y2, x1:x2] > 0
        mask[y1:y2, x1:x2] = sub
    return mask


def mean_rgb_from_mask(img_bgr, mask):
    """Return mean color as (R,G,B) for pixels where mask is True.
    Returns None if mask has no pixels.
    """
    if mask.sum() == 0:
        return None
    vals = img_bgr[mask]  # shape (N,3) in BGR
    mean_bgr = vals.mean(axis=0)
    # convert to ints and to RGB order for display
    mean_rgb = tuple(int(x) for x in mean_bgr[::-1])
    return mean_rgb


def median_rgb_from_mask(img_bgr, mask):
    if mask.sum() == 0:
        return None
    vals = img_bgr[mask]
    med_bgr = np.median(vals, axis=0)
    return tuple(int(x) for x in med_bgr[::-1])


def dominant_rgb_kmeans(img_bgr, mask, k=3):
    """Return dominant color (R,G,B) using k-means (OpenCV) on pixels in mask.
    If there are fewer unique pixels than k or mask empty, falls back to mean.
    """
    if mask.sum() == 0:
        return None
    vals = img_bgr[mask].astype(np.float32)
    if vals.shape[0] < 4:
        return mean_rgb_from_mask(img_bgr, mask)

    # Use OpenCV kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    try:
        compactness, labels, centers = cv2.kmeans(vals, k, None, criteria, 10, flags)
    except Exception:
        # fallback if kmeans fails
        return mean_rgb_from_mask(img_bgr, mask)

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=centers.shape[0])
    dominant_idx = int(np.argmax(counts))
    center_bgr = centers[dominant_idx]
    return tuple(int(x) for x in center_bgr[::-1])


# Build masks for each logical region using the person alpha
face_mask = region_mask_from_box(alpha, face_box)
hair_mask = region_mask_from_box(alpha, hair_box)
torso_mask = region_mask_from_box(alpha, torso_box)
legs_mask = region_mask_from_box(alpha, legs_box)

# ---------------------------------------------------
# Refine face/hair masks to reduce forehead in hair and hair in face
# Use a simple YCrCb skin-color heuristic to resolve pixels in the overlap
# and do small morphological cleanup to remove speckles.
# ---------------------------------------------------
# skin detector thresholds (tweakable)
CR_MIN, CR_MAX = 135, 180
CB_MIN, CB_MAX = 85, 135

# compute YCrCb skin mask on the adjusted image (bgr_adj)
ycrcb = cv2.cvtColor(bgr_adj, cv2.COLOR_BGR2YCrCb)
_, cr, cb = cv2.split(ycrcb)
skin_ycrcb = (cr >= CR_MIN) & (cr <= CR_MAX) & (cb >= CB_MIN) & (cb <= CB_MAX)

# overlapping pixels between hair and face regions
overlap = hair_mask & face_mask
if overlap.any():
    # pixels in overlap that look like skin -> keep in face, remove from hair
    skin_overlap = overlap & skin_ycrcb
    hair_mask[skin_overlap] = False

    # remaining overlap (non-skin) -> likely hair -> remove from face
    non_skin_overlap = overlap & (~skin_ycrcb)
    face_mask[non_skin_overlap] = False

# small morphological cleanups to remove isolated pixels
kernel = np.ones((3,3), np.uint8)
def clean_mask(mask_bool):
    m = (mask_bool.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (m > 0)

face_mask = clean_mask(face_mask)
hair_mask = clean_mask(hair_mask)


# Compute color statistics per region
def stats_for_region(name, img_bgr, mask):
    mean = mean_rgb_from_mask(img_bgr, mask)
    median = median_rgb_from_mask(img_bgr, mask)
    dominant = dominant_rgb_kmeans(img_bgr, mask, k=3)
    # fallback to a single center pixel if everything else fails
    if mean is None:
        # find a non-zero alpha pixel inside the box center (fallback logic)
        ys, xs = np.where(mask)
        if ys.size > 0:
            cy = ys[ys.size//2]
            cx = xs[xs.size//2]
            c_bgr = img_bgr[cy, cx]
            mean = tuple(int(x) for x in c_bgr[::-1])
            median = mean
            dominant = mean
    print(f"{name} mean (R,G,B):", mean, "median:", median, "dominant:", dominant)
    return mean, median, dominant


skin_color, skin_med, skin_dom = stats_for_region('Face', bgr_adj, face_mask)
hair_color, hair_med, hair_dom = stats_for_region('Hair', bgr_adj, hair_mask)
top_color, top_med, top_dom = stats_for_region('Torso', bgr_adj, torso_mask)
bottom_color, bottom_med, bottom_dom = stats_for_region('Legs', bgr_adj, legs_mask)


# ---------------------------------------------------
# Visualization
# ---------------------------------------------------
def save_color_swatch(rgb, filename, size=100):
    """Save a square image showing the RGB color."""
    swatch = np.zeros((size, size, 3), dtype=np.uint8)
    # input is expected as (R,G,B) - convert to BGR for OpenCV
    swatch[:, :] = rgb[::-1]
    cv2.imwrite(filename, swatch)

save_color_swatch(skin_color,   outpath(OUT_SWATCHES, "color_skin.png"))
save_color_swatch(hair_color,   outpath(OUT_SWATCHES, "color_hair.png"))
save_color_swatch(top_color,    outpath(OUT_SWATCHES, "color_top.png"))
save_color_swatch(bottom_color, outpath(OUT_SWATCHES, "color_bottom.png"))
save_color_swatch(skin_med,    outpath(OUT_SWATCHES, "color_skin_median.png"))
save_color_swatch(hair_med,    outpath(OUT_SWATCHES, "color_hair_median.png"))
save_color_swatch(top_med,     outpath(OUT_SWATCHES, "color_top_median.png"))
save_color_swatch(bottom_med,  outpath(OUT_SWATCHES, "color_bottom_median.png"))
save_color_swatch(skin_dom,    outpath(OUT_SWATCHES, "color_skin_dominant.png"))
save_color_swatch(hair_dom,    outpath(OUT_SWATCHES, "color_hair_dominant.png"))
save_color_swatch(top_dom,     outpath(OUT_SWATCHES, "color_top_dominant.png"))
save_color_swatch(bottom_dom,  outpath(OUT_SWATCHES, "color_bottom_dominant.png"))

# Save swatches for computed colors (if available)
if skin_color is not None:
    save_color_swatch(skin_color, outpath(OUT_SWATCHES, "color_skin.png"))
if hair_color is not None:
    save_color_swatch(hair_color, outpath(OUT_SWATCHES, "color_hair.png"))
if top_color is not None:
    save_color_swatch(top_color, outpath(OUT_SWATCHES, "color_top.png"))
if bottom_color is not None:
    save_color_swatch(bottom_color, outpath(OUT_SWATCHES, "color_bottom.png"))

# Save masks as PNGs for debugging
cv2.imwrite(outpath(OUT_MASKS, "mask_face.png"), (face_mask.astype(np.uint8) * 255))
cv2.imwrite(outpath(OUT_MASKS, "mask_hair.png"), (hair_mask.astype(np.uint8) * 255))
cv2.imwrite(outpath(OUT_MASKS, "mask_torso.png"), (torso_mask.astype(np.uint8) * 255))
cv2.imwrite(outpath(OUT_MASKS, "mask_legs.png"), (legs_mask.astype(np.uint8) * 255))

# Create a visualization with blended region colors
vis = bgr.copy()
overlay = vis.copy()
def apply_region_overlay(overlay_img, mask, rgb_color):
    if rgb_color is None:
        return
    bgr_color = tuple(int(x) for x in rgb_color[::-1])
    overlay_img[mask] = bgr_color

apply_region_overlay(overlay, face_mask, skin_color)
apply_region_overlay(overlay, hair_mask, hair_color)
apply_region_overlay(overlay, torso_mask, top_color)
apply_region_overlay(overlay, legs_mask, bottom_color)

# blend only where any region exists
any_mask = (face_mask | hair_mask | torso_mask | legs_mask)
# blend only where any region exists
vis_blend = vis.copy()
vis_blend[any_mask] = cv2.addWeighted(vis[any_mask], 0.6, overlay[any_mask], 0.4, 0)
cv2.imwrite(outpath(OUT_OVERLAYS, "color_region_overlay.jpg"), vis_blend)
# # Also save the original and CLAHE-normalized images for preview
# cv2.imwrite(outpath(OUT_IMAGES, "input_original.png"), bgr)
# cv2.imwrite(outpath(OUT_IMAGES, "input_clahe.png"), bgr_norm)

# # Create a simple 2x2 preview grid: original | CLAHE \n adjusted | overlay
# def make_preview_grid(imgs, thumb_size=(400,400)):
#     # imgs: list of 4 BGR images
#     resized = []
#     for im in imgs:
#         try:
#             r = cv2.resize(im, thumb_size, interpolation=cv2.INTER_AREA)
#         except Exception:
#             # if resize fails (e.g., None), create a blank image
#             r = np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
#         resized.append(r)

#     top = np.hstack((resized[0], resized[1]))
#     bottom = np.hstack((resized[2], resized[3]))
#     grid = np.vstack((top, bottom))
#     return grid

# preview_imgs = [bgr, bgr_norm, bgr_adj, vis_blend]
# preview_grid = make_preview_grid(preview_imgs, thumb_size=(400,400))
# cv2.imwrite(outpath(OUT_PREV, "preview_grid.jpg"), preview_grid)

# # Write a tiny HTML preview that shows the grid and links to component images
# html = f"""<!doctype html>
# <html>
# <head><meta charset='utf-8'><title>Image preview</title></head>
# <body>
# <h2>Preview: original | CLAHE</h2>
# <h2>adjusted | overlay</h2>
# <img src='preview_grid.jpg' alt='preview grid' style='max-width:100%;height:auto;'>
# <p>Individual images:</p>
# <ul>
#   <li><a href='input_original.png'>input_original.png</a></li>
#   <li><a href='input_clahe.png'>input_clahe.png</a></li>
#   <li><a href='input_adjusted.png'>input_adjusted.png</a></li>
#   <li><a href='color_region_overlay.jpg'>color_region_overlay.jpg</a></li>
#   <li><a href='mask_face.png'>mask_face.png</a></li>
#   <li><a href='mask_torso.png'>mask_torso.png</a></li>
#   <li><a href='mask_hair.png'>mask_hair.png</a></li>
#   <li><a href='mask_legs.png'>mask_legs.png</a></li>
# </ul>
# </body>
# </html>
# """
# def img_to_data_uri(img, fmt='.png'):
#         # encode image (BGR) to memory buffer then base64
#         ok, buf = cv2.imencode(fmt, img)
#         if not ok:
#                 return ''
#         b64 = base64.b64encode(buf.tobytes()).decode('ascii')
#         mime = 'image/png' if fmt.lower().endswith('png') or fmt == '.png' else 'image/jpeg'
#         return f'data:{mime};base64,{b64}'

# # prepare inline data URIs for the four quadrant images
# data_original = img_to_data_uri(bgr, fmt='.png')
# data_clahe = img_to_data_uri(bgr_norm, fmt='.png')
# data_adjusted = img_to_data_uri(bgr_adj, fmt='.png')
# data_overlay = img_to_data_uri(vis_blend, fmt='.jpg')

# # prepare paths relative to preview.html which lives in OUTPUT_DIR
# rel_images = os.path.relpath(OUT_IMAGES, OUTPUT_DIR)
# rel_masks = os.path.relpath(OUT_MASKS, OUTPUT_DIR)
# rel_swatches = os.path.relpath(OUT_SWATCHES, OUTPUT_DIR)
# rel_overlays = os.path.relpath(OUT_OVERLAYS, OUTPUT_DIR)
# rel_previews = os.path.relpath(OUT_PREV, OUTPUT_DIR)

# html = f"""<!doctype html>
# <html>
# <head>
#     <meta charset='utf-8'>
#     <title>Image preview</title>
#     <style>
#         body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
#         .grid {{ display: grid; grid-template-columns: 1fr 1fr; grid-gap: 12px; max-width: 820px; }}
#         .cell {{ border: 1px solid #ddd; padding: 8px; background: #fafafa; text-align: center; }}
#         .cell h3 {{ margin: 6px 0; font-size: 16px; }}
#         .thumb {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
#         .actions {{ margin-top: 6px; }}
#         .btn {{ display: inline-block; padding: 6px 10px; margin: 0 6px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; font-size: 13px; }}
#         .btn.secondary {{ background: #6c757d; }}
#     </style>
# </head>
# <body>
#     <h1>Image preview</h1>
#     <div class='grid'>
#         <div class='cell'>
#             <h3>Original</h3>
#             <img class='thumb' src='{data_original}' alt='original'>
#             <div class='actions'>
#                 <a class='btn' href='{rel_images}/input_original.png' download>Download PNG</a>
#                 <a class='btn secondary' href='{data_original}' target='_blank'>Open inline</a>
#             </div>
#         </div>

#         <div class='cell'>
#             <h3>CLAHE Normalized</h3>
#             <img class='thumb' src='{data_clahe}' alt='clahe'>
#             <div class='actions'>
#                 <a class='btn' href='{rel_images}/input_clahe.png' download>Download PNG</a>
#                 <a class='btn secondary' href='{data_clahe}' target='_blank'>Open inline</a>
#             </div>
#         </div>

#         <div class='cell'>
#             <h3>Adjusted (brightness / saturation)</h3>
#             <img class='thumb' src='{data_adjusted}' alt='adjusted'>
#             <div class='actions'>
#                 <a class='btn' href='{rel_images}/input_adjusted.png' download>Download PNG</a>
#                 <a class='btn secondary' href='{data_adjusted}' target='_blank'>Open inline</a>
#             </div>
#         </div>

#         <div class='cell'>
#             <h3>Region overlay</h3>
#             <img class='thumb' src='{data_overlay}' alt='overlay'>
#             <div class='actions'>
#                 <a class='btn' href='{rel_overlays}/color_region_overlay.jpg' download>Download JPG</a>
#                 <a class='btn secondary' href='{data_overlay}' target='_blank'>Open inline</a>
#             </div>
#         </div>
#     </div>

#     <h4>Other artifacts</h4>
#     <ul>
#         <li><a href='{rel_previews}/preview_grid.jpg'>preview_grid.jpg</a></li>
#         <li><a href='{rel_masks}/mask_face.png'>mask_face.png</a></li>
#         <li><a href='{rel_masks}/mask_hair.png'>mask_hair.png</a></li>
#         <li><a href='{rel_masks}/mask_torso.png'>mask_torso.png</a></li>
#         <li><a href='{rel_masks}/mask_legs.png'>mask_legs.png</a></li>
#     </ul>
# </body>
# </html>
# """

# with open(outpath(OUTPUT_DIR, "preview.html"), "w", encoding="utf-8") as f:
#     f.write(html)

print(f"Completed: saved outputs under '{OUTPUT_DIR}/' (masks, swatches, overlays, images, previews). Preview available at {outpath(OUTPUT_DIR,'preview.html')}")
