from rembg import remove
from PIL import Image

# Load your image
input_path = "./input_image/IMG_1290.JPG"
output_path = "./input_image/person_cutout.png"

input_image = Image.open(input_path)

# Remove background
output_image = remove(input_image)

# Save with transparency
output_image.save(output_path)
