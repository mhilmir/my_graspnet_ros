import cv2
import numpy as np

# Load the RGB image
rgb_image = cv2.imread('graspnet_input/color_ros.png')  # Replace with your image path

# Load the grayscale mask and convert to float32
mask_gray = cv2.imread('graspnet_input/depth_ros.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Normalize if max is 1 or less
if mask_gray.max() <= 1.0:
    mask_gray *= 255.0  # Scale up to 0–255

# Convert to uint8
mask_gray = mask_gray.astype(np.uint8)

# Resize mask to match RGB size if needed
if mask_gray.shape != rgb_image.shape[:2]:
    mask_gray = cv2.resize(mask_gray, (rgb_image.shape[1], rgb_image.shape[0]))

# Create a 3-channel mask
mask_3ch = cv2.merge([mask_gray] * 3)

# Apply the mask using bitwise_and
result = cv2.bitwise_and(rgb_image, mask_3ch)

# Save or display the result
cv2.imwrite('combined_output.png', result)
cv2.imshow('Combined Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()