from PIL import Image
import numpy as np

def extract_last_channel(input_path, output_path):
    # Open the image
    img = Image.open(input_path)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Extract the last channel
    last_channel = img_array[:, :, -1]
    
    # Convert the last channel to an image
    last_channel_img = Image.fromarray(last_channel)
    
    # Save the image
    last_channel_img.save(output_path)

if __name__ == "__main__":
    input_path = "/home/usrs/taniuchi/workspace/zatta/20241219_中間発表用_MFNet/00292D.png"  # Replace with your input image path
    output_path = "/home/usrs/taniuchi/workspace/zatta/20241219_中間発表用_MFNet/00292D_1ch.jpg"  # Replace with your desired output image path
    extract_last_channel(input_path, output_path)