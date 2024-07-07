import os

import shutil
import random
#use this for annoying datasets that aren't presplit
def split_data(source_dir, train_dir, test_dir, split_ratio=0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Iterate through each subfolder in the source directory
    for subfolder in os.listdir(source_dir):
        subfolder_path = os.path.join(source_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Create corresponding subfolders in train and test directories
            train_subfolder = os.path.join(train_dir, subfolder)
            test_subfolder = os.path.join(test_dir, subfolder)
            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)

            # List all files in the subfolder
            files = os.listdir(subfolder_path)
            random.shuffle(files)  # Shuffle files randomly

            # Split files into train and test
            split_index = int(len(files) * split_ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]

            # Move files to train and test subfolders
            for file in train_files:
                shutil.move(os.path.join(subfolder_path, file), os.path.join(train_subfolder, file))
            for file in test_files:
                shutil.move(os.path.join(subfolder_path, file), os.path.join(test_subfolder, file))

source_folder = "/home/yeoman/research/NHF"
train_folder = '/home/yeoman/MIND/models_vit/data/train'
test_folder = '/home/yeoman/MIND/models_vit/data/test'

split_data(source_folder, train_folder, test_folder)