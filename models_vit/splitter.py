import os
from torchvision import datasets, transforms
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

# split_data(source_folder, train_folder, test_folder)

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    

walk_through_dir(train_folder)
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])
train_data = datasets.ImageFolder(root=train_folder, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_folder, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
class_names = train_data.classes
class_dict = train_data.class_to_idx
print(class_dict)