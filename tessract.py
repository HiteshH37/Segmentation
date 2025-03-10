import pytesseract
from PIL import Image

# Paths to images
nor_image_path = "C:/Users/HITESH/Documents/Programmes written in Python/Machine learning/OCR/Preprosing and segmentation/preprocessed.png"


# Open images
nor_image = Image.open(nor_image_path)


# Extract text from image
nor_text = pytesseract.image_to_string(nor_image, lang='kan')

# Save to a file
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(nor_text)


print("Kannada text extracted and saved to output.txt")
