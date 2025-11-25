from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)

# Start the camera
picam2.start()

# Wait for the camera to warm up
time.sleep(2)

# Capture an image
timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"./image_{timestamp}.jpg"
picam2.capture_file(filename)

print(f"Picture saved as {filename}")

# Close the camera
picam2.close()
