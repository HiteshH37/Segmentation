import cv2
import numpy as np
import os

# Directories
INPUT_IMAGE = "Test.jpeg"  # Path to the input image
CHARACTERS_DIR = "./characters"  # Folder for saving cropped characters

# Create the output directory if it doesn't exist
os.makedirs(CHARACTERS_DIR, exist_ok=True)

# Step 1: Load and Preprocess Image (Convert to Grayscale & Binarize)
def preprocess_image(image_path):
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read original grayscale image
    _, binary_img = cv2.threshold(original_img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binarize
    
    # Apply dilation to improve character segmentation
    kernel = np.ones((1, 2), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)

    return original_img, binary_img  # Return both images

# Step 2: Detect and Crop Individual Characters
def crop_characters(original_img, binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # Filter small contours (noise)
            char = original_img[y:y + h, x:x + w]  # Crop from the original image
            characters.append((x, y, w, h, char))  # Store bounding box info for sorting

    # Sort by y first (to process top lines first), then by x within each line
    characters.sort(key=lambda item: item[1])  # Sort by y-coordinate

    # Group into lines based on proximity of y-coordinates
    lines = []
    current_line = []
    line_threshold = 15  # Adjust based on character height to detect line breaks

    for i, char in enumerate(characters):
        if i == 0:
            current_line.append(char)
            continue

        prev_char = characters[i - 1]
        if abs(char[1] - prev_char[1]) < line_threshold:
            current_line.append(char)
        else:
            lines.append(current_line)
            current_line = [char]

    if current_line:
        lines.append(current_line)

    # Sort each line left-to-right
    for line in lines:
        line.sort(key=lambda item: item[0])

    # Flatten sorted characters
    sorted_characters = [char[4] for line in lines for char in line]

    return sorted_characters

# Step 3: Save Characters as Separate Images (Original Image Cropped)
def save_characters(characters):
    for idx, char in enumerate(characters):
        char_path = os.path.join(CHARACTERS_DIR, f"char_{idx}.png")
        cv2.imwrite(char_path, char)

# Main Script
if os.path.exists(INPUT_IMAGE):
    # Preprocess the image
    original_image, preprocessed_image = preprocess_image(INPUT_IMAGE)

    # Detect and crop characters from the original image
    characters = crop_characters(original_image, preprocessed_image)
    print(f"Detected {len(characters)} characters in the image.")

    # Save each character (from original image)
    save_characters(characters)
    print(f"All characters have been saved in '{CHARACTERS_DIR}'.")
else:
    print(f"Input image '{INPUT_IMAGE}' not found. Please provide a valid image path.")
