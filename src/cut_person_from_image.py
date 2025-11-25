from rembg import remove
from PIL import Image

# Load your image
input_path = "/Users/kayvon/Projects/TakeAPicture/IMG_1290.jpg"
output_path = "/Users/kayvon/Projects/TakeAPicture/person_cutout.png"

input_image = Image.open(input_path)

# Remove background
output_image = remove(input_image)

# Save with transparency
output_image.save(output_path)
