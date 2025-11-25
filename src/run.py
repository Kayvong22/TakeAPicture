# 1. Create template (using the annotation tool)
# - Load sprite.png
# - Mark skin, hair, clothes, outlines
# - Save as character_template.json

# 2. Use in your project
import os
from src.sprite_renderer import SpriteRenderer

for path_sprite_template in os.listdir("./sprites_template/"):
    if not path_sprite_template.endswith("_template.json"):
        continue
    path_sprite_template = os.path.join("./sprites_template/", path_sprite_template)
    print("Processing:", path_sprite_template)

    name_sprite = path_sprite_template.split("/")[-1].replace("_template.json", "")

    renderer = SpriteRenderer(path_sprite_template)

    # Your input colors
    colors = {
        "skin": "#b09080",
        "hair": "#1d1916",
        "top": "#695d46",
        "bottom": "#464033"
    }

    # Pass the original sprite path to preserve unclassified pixels
    renderer.save_render(
        colors, 
        f"./sprites_final/{name_sprite}.png", 
        scale=1,
        original_image_path=f"./sprites_original/{name_sprite}.png"
    )
