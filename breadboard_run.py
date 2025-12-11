import os
import random
import argparse
import time
from pathlib import Path
from PIL import Image
from gpiozero import LED, Button
from signal import pause

# Attempt to import the renderer from the same folder
try:
    from sprite_renderer import SpriteRenderer
except Exception:
    # If running from project root, allow import via src package style
    from src.sprite_renderer import SpriteRenderer


def find_repo_root():
    """Return a sensible project root.

    Prefer the current working directory if it looks like the project (contains
    expected marker files/folders). Otherwise try to locate a repository root
    relative to this file. This avoids cases where `__file__` points into the
    system python lib (e.g. `/usr/lib/python3.13/...`) when the module was
    installed or run in an unusual way.
    """
    cwd = Path.cwd().resolve()
    markers = ["sprites_template", "src", "breadboard_run.py", ".git", "README.md"]

    # If current working directory contains any repo markers, prefer it.
    try:
        if any((cwd / m).exists() for m in markers):
            return cwd
    except Exception:
        # In some restricted environments Path.cwd() may fail; we'll ignore and fallthrough
        pass

    # Fallback: resolve relative to this file and walk up looking for markers
    try:
        base = Path(__file__).resolve()
    except Exception:
        # If __file__ isn't available, return cwd
        return cwd

    p = base
    # Walk upwards up to filesystem root
    while True:
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent

    # Final fallback to cwd
    return cwd


ROOT = find_repo_root()
SWATCH_DIR = ROOT / "outputs" / "swatches"
TEMPLATE_DIR = ROOT / "sprites_template"
ORIGINAL_DIR = ROOT / "sprites_original"
OUTPUT_DIR = ROOT / "sprites_final"

led1 = LED(17)
led2 = LED(27)
led3 = LED(22)
button = Button(26)

led1.off()
led2.off()
led3.off()

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


def choose_template(template_choice: str = None):
    """Return a Path to the chosen template.

    If `template_choice` is None, pick a random template. Otherwise accept:
    - a path to an existing file
    - a filename (with or without `_template.json`)
    - a substring to match a template filename
    """
    templates = find_templates()
    if not templates:
        raise FileNotFoundError(f"No templates found in {TEMPLATE_DIR}")

    if not template_choice:
        return random.choice(templates)

    # If user passed a path that exists, use it
    choice_path = Path(template_choice)
    if choice_path.exists():
        return choice_path

    # Direct filename in templates dir
    candidate = TEMPLATE_DIR / template_choice
    if candidate.exists():
        return candidate

    # Add suffix and try
    candidate2 = TEMPLATE_DIR / f"{template_choice}_template.json"
    if candidate2.exists():
        return candidate2

    # Try substring match
    for p in templates:
        if template_choice in p.name:
            return p

    raise FileNotFoundError(f"No template matching '{template_choice}' in {TEMPLATE_DIR}")


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


def main(template_choice: str = None):
    ensure_output_dir()
    # Ensure LEDs are always turned off on exit/error
    try:
        led1.on()

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

        # Choose a template (random if not provided)
        tmpl = choose_template(template_choice)
        print(f"Selected template: {tmpl.name}")
        led1.off()
        led2.on()

        # Try to locate original sprite (optional)
        original = matching_original_for(tmpl)

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
        led2.off()
        led3.on()
        time.sleep(10)
    finally:
        # Ensure all LEDs are turned off even if something fails
        try:
            led3.off()
        except Exception:
            pass
        try:
            led2.off()
        except Exception:
            pass
        try:
            led1.off()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a sprite from swatches and a template")
    parser.add_argument("--template", "-t", help="Template to use (path, filename, or substring). If omitted, a random template is chosen.")
    args = parser.parse_args()
    # If a template was provided on the command line, run once and exit
    if args.template:
        main(template_choice=args.template)
    else:
        # No template provided: set button to trigger a run using a random template
        button.when_pressed = lambda: main(template_choice=None)
        # Wait for button presses (keep process alive)
        pause()
