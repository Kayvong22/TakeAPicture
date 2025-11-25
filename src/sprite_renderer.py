from PIL import Image
import json
from typing import Dict, Tuple

class SpriteRenderer:
    """
    Renders pixel art sprites from templates with custom colors.
    Preserves shading and highlights based on original pixel brightness.
    """
    
    def __init__(self, template_path: str):
        """
        Load a sprite template from JSON file.
        
        Args:
            template_path: Path to the template JSON file
        """
        with open(template_path, 'r') as f:
            self.template = json.load(f)
            
        self.width = self.template["width"]
        self.height = self.template["height"]
        
        # Build lookup dictionary for faster rendering
        self.pixel_map = {}
        for pixel_data in self.template["pixels"]:
            x = pixel_data["x"]
            y = pixel_data["y"]
            self.pixel_map[(x, y)] = {
                "category": pixel_data["category"],
                "original_color": (
                    pixel_data["original_color"]["r"],
                    pixel_data["original_color"]["g"],
                    pixel_data["original_color"]["b"]
                )
            }
    
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def calculate_brightness(self, color: Tuple[int, int, int]) -> float:
        """
        Calculate perceived brightness of a color (0.0 to 1.0).
        Uses standard luminance formula.
        """
        r, g, b = color
        # Perceived luminance formula
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    def apply_shading(self, target_color: Tuple[int, int, int], 
                     original_brightness: float) -> Tuple[int, int, int]:
        """
        Apply brightness-based shading to a target color.
        
        Args:
            target_color: The base color to shade (RGB tuple)
            original_brightness: Brightness of original pixel (0.0 to 1.0)
            
        Returns:
            Shaded RGB color tuple
        """
        # Use a curve to preserve both shadows and highlights
        # This makes dark areas darker and light areas lighter
        if original_brightness < 0.5:
            # Shadows: darken proportionally
            factor = original_brightness * 2
        else:
            # Highlights: lighten with emphasis
            factor = 0.5 + (original_brightness - 0.5) * 1.5
        
        # Apply factor and clamp to valid range
        shaded = tuple(
            int(min(255, max(0, c * factor)))
            for c in target_color
        )
        
        return shaded
    
    def render(self, colors: Dict[str, str], background_color: str = None, 
               original_image_path: str = None) -> Image.Image:
        """
        Render the sprite with custom colors.
        
        Args:
            colors: Dictionary mapping categories to hex colors
                   Example: {
                       "skin": "#FFC0A8",
                       "hair": "#8B4513",
                       "top": "#FF0000",
                       "bottom": "#0000FF"
                   }
            background_color: Optional hex color for background (default: transparent)
            original_image_path: Optional path to original sprite to preserve unclassified pixels
            
        Returns:
            PIL Image object of rendered sprite
        """
        # Create output image
        if background_color:
            bg_rgb = self.hex_to_rgb(background_color)
            result = Image.new("RGB", (self.width, self.height), bg_rgb)
        else:
            result = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        
        # If original image provided, use it as base to preserve unclassified pixels
        if original_image_path:
            original_img = Image.open(original_image_path).convert("RGBA")
            for y in range(self.height):
                for x in range(self.width):
                    pixel = original_img.getpixel((x, y))
                    if pixel[3] > 0:  # If not fully transparent
                        if background_color:
                            result.putpixel((x, y), pixel[:3])
                        else:
                            result.putpixel((x, y), pixel)
        
        # Convert input colors to RGB
        color_rgb = {cat: self.hex_to_rgb(hex_col) for cat, hex_col in colors.items()}
        
        # Render each categorized pixel (overrides original pixels)
        for (x, y), pixel_info in self.pixel_map.items():
            category = pixel_info["category"]
            original_color = pixel_info["original_color"]
            
            if category == "outline":
                # Outlines stay dark, preserving original variation
                brightness = self.calculate_brightness(original_color)
                outline_val = int(brightness * 80)  # Dark but not pure black
                final_color = (outline_val, outline_val, outline_val)
            else:
                # Apply shading based on original brightness
                if category in color_rgb:
                    target_color = color_rgb[category]
                    original_brightness = self.calculate_brightness(original_color)
                    final_color = self.apply_shading(target_color, original_brightness)
                else:
                    # Category not provided, use original color
                    final_color = original_color
            
            if background_color:
                result.putpixel((x, y), final_color)
            else:
                result.putpixel((x, y), final_color + (255,))
        
        return result
    
    def render_multiple(self, color_variations: list, background_color: str = None,
                       original_image_path: str = None) -> list:
        """
        Render multiple sprites with different color combinations.
        
        Args:
            color_variations: List of color dictionaries
            background_color: Optional background color
            original_image_path: Optional path to original sprite
            
        Returns:
            List of PIL Image objects
        """
        return [self.render(colors, background_color, original_image_path) 
                for colors in color_variations]
    
    def save_render(self, colors: Dict[str, str], output_path: str, 
                    scale: int = 1, background_color: str = None,
                    original_image_path: str = None):
        """
        Render and save sprite to file.
        
        Args:
            colors: Color dictionary
            output_path: Path to save the image
            scale: Scaling factor (1 = original size)
            background_color: Optional background color
            original_image_path: Optional path to original sprite to preserve unclassified pixels
        """
        sprite = self.render(colors, background_color, original_image_path)
        
        if scale > 1:
            new_size = (self.width * scale, self.height * scale)
            sprite = sprite.resize(new_size, Image.NEAREST)
        
        sprite.save(output_path)
