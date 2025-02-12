from PIL import Image
import os

def convert_rgba_to_rgb(input_path, output_path):
    # Open the input image
    with Image.open(input_path) as img:
        # Ensure the image is in RGBA mode
        img = img.convert("RGBA")
        
        # Create a new image with an RGB mode
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Paste using alpha channel as mask
        
        # Save the new image
        rgb_img.save(output_path, "PNG")

if __name__ == "__main__":
    input_path = "/home/usrs/taniuchi/workspace/zatta/20241219_中間発表用_MFNet/00292D.png"  # Replace with your input image path
    output_path = "/home/usrs/taniuchi/workspace/zatta/20241219_中間発表用_MFNet/00292D_3ch.jpg"  # Replace with your desired output image path
    convert_rgba_to_rgb(input_path, output_path)