"""
Render a random sprite using color analysis swatches saved in `outputs/swatches`.

Reads swatch PNGs (prefers plain -> median -> dominant), selects a random
template from `sprites_template/`, optionally uses the matching original
in `sprites_original/` if present, then renders and saves to `sprites_final/`.

Usage (from project root):
    python3 src/render_your_sprite.py

Optional environment variables / args can be added later.
"""
import os
import random
from pathlib import Path
from PIL import Image

# Attempt to import the renderer from the same folder
try:
    from sprite_renderer import SpriteRenderer
except Exception:
    # If running from project root, allow import via src package style
    from src.sprite_renderer import SpriteRenderer


ROOT = Path(__file__).resolve().parents[1]
SWATCH_DIR = ROOT / "outputs" / "swatches"
TEMPLATE_DIR = ROOT / "sprites_template"
ORIGINAL_DIR = ROOT / "sprites_original"
OUTPUT_DIR = ROOT / "sprites_final"


def swatch_candidates(category: str):
    """Return candidate filenames (in order of preference) for a category."""
    base = f"color_{category}"
    return [f"{base}.png", f"{base}_median.png", f"{base}_dominant.png"]


def read_swatch_hex(category: str):
    """Read a swatch image and return its center pixel as a hex color string.

    Tries plain -> median -> dominant. Returns None if no swatch found.
    """
    for fname in swatch_candidates(category):
        path = SWATCH_DIR / fname
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                # sample center pixel; swatches are expected to be solid
                px = img.getpixel((w // 2, h // 2))
                return rgb_to_hex(px)
            except Exception:
                continue
    return None


def rgb_to_hex(rgb_tuple):
    return "#%02x%02x%02x" % rgb_tuple


def find_templates():
    """List template JSON files in the template directory."""
    if not TEMPLATE_DIR.exists():
        return []
    return [p for p in TEMPLATE_DIR.iterdir() if p.name.endswith("_template.json")]


def choose_random_template():
    templates = find_templates()
    if not templates:
        raise FileNotFoundError(f"No templates found in {TEMPLATE_DIR}")
    return random.choice(templates)


def matching_original_for(template_path: Path):
    """Given a template file path, return path to matching original image if present."""
    name = template_path.name.replace("_template.json", "")
    # expect PNG named the same in sprites_original
    candidate = ORIGINAL_DIR / f"{name}.png"
    if candidate.exists():
        return str(candidate)
    return None


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_output_dir()

    # Read colors from swatches
    categories = ["skin", "hair", "top", "bottom"]
    colors = {}
    for c in categories:
        hexcol = read_swatch_hex(c)
        if hexcol is None:
            print(f"Warning: no swatch found for '{c}' in {SWATCH_DIR}; using fallback gray")
            hexcol = "#888888"
        colors[c] = hexcol

    print("Using colors:")
    for k, v in colors.items():
        print(f"  {k}: {v}")

    # Choose a random template
    tmpl = choose_random_template()
    print(f"Selected template: {tmpl.name}")

    # Try to locate original sprite (optional)
    original = matching_original_for(tmpl)
    if original:
        print(f"Original sprite to preserve unclassified pixels: {original}")
    else:
        print("No matching original sprite found; rendering without original image.")

    # Render
    renderer = SpriteRenderer(str(tmpl))

    name = tmpl.name.replace("_template.json", "")
    out_path = OUTPUT_DIR / f"{name}.png"

    try:
        renderer.save_render(colors, str(out_path), scale=1, background_color=None, original_image_path=original)
    except TypeError:
        # Older renderer signature may not accept background_color first; try without it
        renderer.save_render(colors, str(out_path), scale=1, original_image_path=original)

    print(f"Saved rendered sprite to: {out_path}")


if __name__ == "__main__":
    main()
