from rembg import remove
from PIL import Image

def CutPersonFromImage(input_path: str, output_path: str):
    """Cut out the person from the input image and save with transparency."""
    input_image = Image.open(input_path)
    # Remove background
    output_image = remove(input_image)
    # Save with transparency
    output_image.save(output_path)
