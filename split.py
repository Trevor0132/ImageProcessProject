'''
FilePath: \project_1\classfy.py
Brief: 
Author: Trevor(wuchenfeng0132@qq.com)
Date: 2024-05-09 20:32:29
'''
import os
import random
import shutil

def split_images(input_folder, output_folder1, output_folder2, split_ratio):
    # Create the output folders if they don't exist
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    # Get a list of all image files in the input folder
    image_files = [file for file in os.listdir(input_folder) if file.endswith('.tiff')]

    # Shuffle the list of image files randomly
    random.shuffle(image_files)

    # Calculate the split index based on the split ratio
    split_index = int(len(image_files) * split_ratio)

    # Split the image files into two lists based on the split index
    images1 = image_files[:split_index]
    images2 = image_files[split_index:]

    # Move images to the output folders
    for image in images1:
        input_path = os.path.join(input_folder, image)
        output_path = os.path.join(output_folder1, image)
        shutil.move(input_path, output_path)

    for image in images2:
        input_path = os.path.join(input_folder, image)
        output_path = os.path.join(output_folder2, image)
        shutil.move(input_path, output_path)

    print(f"Images split into {output_folder1} and {output_folder2} with ratio {split_ratio}.")


# Split TIFF images in the input folder into two output folders with ratio 8:2

# Specify the input folder and output folders
input_folder = r"src data path"
output_folder1 = r"train data path"
output_folder2 = r"test data path"

# Split images in the input folder into two output folders with ratio 8:2
split_images(input_folder, output_folder1, output_folder2, split_ratio=0.8)