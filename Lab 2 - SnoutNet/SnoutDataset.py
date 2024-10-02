import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import decode_image, ImageReadMode
from snoutTransforms import ToTensor, RescaleImage
import glob
from typing import Tuple, Any, Optional, Union, List
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import pandas as pd
import os
from skimage import io
import numpy as np
from tqdm import tqdm
from PIL import Image

DECODE_MODE = ImageReadMode.RGB
IMAGE_SIZE = (227, 227)

def read_file(label_path: Union[str, Path])->Tuple[List, np.ndarray]:
    """
    Reads a given text file and extracts image paths and coordinate pairs.

    Args:
        label_path (Union[str, Path]): path to the .txt file with the required information.

    Returns:
        Tuple[List, np.ndarray]: List of image paths, numpy array of respective coordinates.
    """
    img_paths = []
    snout_tups = []
    
    with open(label_path, "r") as file:
        line = file.readline()
        while line:
            img_path, snout_center = line.rstrip().split(",", 1)
            snout_center = ast.literal_eval(ast.literal_eval(snout_center)) # need the nested ast.literal_eval() to break out of the double quotes (because snout center is technically a string within a string (line))
            line = file.readline()

            try:
                with Image.open(os.path.join('images/', img_path)) as img:  # using PIL.Image.open() and .verify() to check the integrity of images, using skimage to actually read
                    img.verify()
                    img_paths.append(img_path)
                    snout_tups.append(snout_center)
            except (IOError, SyntaxError) as e:
                print(f'Corrupted file: {img_path} \n Error {e}')

    snout_tups = np.array(snout_tups, dtype=float).reshape(-1,2)
    print(f"Len img_paths: {len(img_paths)} : Len tuples: {len(snout_tups)}")
    return img_paths, snout_tups

# shows the landmarks for a single batch
def show_landmarks(image: Union[torch.Tensor, np.ndarray], center):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(center[0], center[1], s=10, marker='*', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

# will show images and centers overlaid on top for a given batch
def show_batch(batch: dict):
    batch_images, batch_center = batch['image'], batch['center']
    batch_size = len(batch_images)
    im_size = batch_images.size(2)
    grid_border_size = 2
   
    grid = utils.make_grid(batch_images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
  #      print(f"center: {batch_center[i]}")
        plt.scatter(batch_center[i, 0].numpy() + i * im_size + (i+1) * grid_border_size, 
                    batch_center[i, 1].numpy() + grid_border_size,
                    s=10, marker='*', c='r'
                    )

# BEGIN CLASS DEFINITION

class SnoutDataset(Dataset):
    def __init__(self, label_path: Union[str, Path], image_folder: Union[str, Path], transform=None)->None:
        self.transform = transform
        self.image_paths, self.snout_tuples = read_file(label_path)
        self.image_folder = image_folder

    def __len__(self)->int:
        return len(self.image_paths)
    
    # implement transform to the labels here
    def __getitem__(self, idx: Any)->dict:
        """
        Fetch a single item given an index

        Args:
            idx (Any): index of the desired item

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: returns a tensor representation of the image and tuple of coordinates describing the nose

        """
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):         # numpy and torch have the same method to cast to a list
            idx = idx.tolist()
        
        path = os.path.join(self.image_folder, self.image_paths[idx])
        image = io.imread(path)                                         # this should be a tensor (https://pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image)
        snout_center = self.snout_tuples[idx]
        sample = {'image':image, 'center':snout_center}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    label_path = Path("train_noses.txt")
    image_folder = Path("images")

    transform_pipeline = transforms.Compose([RescaleImage(IMAGE_SIZE), ToTensor()])

    dataset = SnoutDataset(label_path=label_path, image_folder=image_folder, transform=transform_pipeline)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    for i, batch in enumerate(dataloader):

        if i == 25:
            print(f"Batch Image size: {batch['image'].size()} \n Batch Center size: {batch['center'].size()}")

            plt.figure(figsize=(10,10))
            show_batch(batch)
            plt.axis('off')
            plt.ioff()
            plt.title(f"Batch {i} from dataloader")
            plt.show()
            break
    
