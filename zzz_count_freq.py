import numpy as np
import os
import cv2
import tqdm

def count_pixel_values(image):
    # Initialize a dictionary to store the counts of each pixel value
    pixel_counts = {i: 0 for i in range(9)}
    
    # Flatten the image array to make it easier to count pixel values
    flattened_image = image.flatten()
    
    # Count the occurrences of each pixel value
    for pixel in flattened_image:
        if pixel in pixel_counts:
            pixel_counts[pixel] += 1
    
    return pixel_counts

def count_pixel_values_in_directory(directory_path):
    # Initialize a dictionary to store the total counts of each pixel value
    total_pixel_counts = {i: 0 for i in range(9)}
    
    # Iterate over each file in the directory
    for filename in tqdm.tqdm(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        
        # Read the image file
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Get the pixel counts for the current image
            pixel_counts = count_pixel_values(image)
            
            # Add the counts to the total counts
            for pixel_value, count in pixel_counts.items():
                total_pixel_counts[pixel_value] += count
    
    # Print the total counts of each pixel value
    for pixel_value, count in total_pixel_counts.items():
        print(f"Pixel value {pixel_value}: {count} times")

# Example usage
if __name__ == "__main__":
    directory_path = "/home/usrs/taniuchi/workspace/datasets/ir_seg_dataset/labels"
    count_pixel_values_in_directory(directory_path)