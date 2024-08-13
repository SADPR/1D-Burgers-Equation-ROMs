from pdf2image import convert_from_path
import os

# Set the path to your PDF file
pdf_path = 'StrongForm.pdf'

# Create a directory to save PNG images
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# Convert PDF to images
images = convert_from_path(pdf_path)

# Save each page as a PNG file
for i, image in enumerate(images):
    image_path = os.path.join(output_folder, f'StrongForm.png')
    image.save(image_path, 'PNG')

print(f'PDF converted to PNG images and saved in "{output_folder}" directory.')
