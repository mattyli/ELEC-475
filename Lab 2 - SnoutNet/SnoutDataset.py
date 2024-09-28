import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, transforms
from torchvision.io import decode_image, ImageReadMode
import glob
from typing import Tuple, Any, Optional, Union, List
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import pandas as pd
import os
from skimage import io, transform
import numpy as np

DECODE_MODE = ImageReadMode.RGB

def read_file(filepath: Union[str, Path])->Tuple[List, List]:
    img_paths = []
    snout_tups = []
    with open(filepath, "r") as file:
        line = file.readline()
        while line:
            img_path, snout_center = line.rstrip().split(",", 1)
            snout_center = ast.literal_eval(ast.literal_eval(snout_center)) # need the nested ast.literal_eval() to break out of the double quotes (because snout center is technically a string within a string (line))
            line = file.readline()
            img_paths.append(img_path)
            snout_tups.append(snout_center)
    return img_paths, snout_tups

def show_landmarks(image: Union[torch.Tensor, np.ndarray], center):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(center[0], center[1], s=10, marker='*', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


class SnoutDataset(Dataset):
    def __init__(self, infopath: Union[str, Path], datapath: Union[str, Path], transform=None)->None:
        self.transform = transform
        self.image_paths, self.snout_tuples = read_file(infopath)
        self.datapath = datapath

        # convert the images to tensors
    """
    input from files are tuple(filepath, tuple(x,y))
    target is stored as a text file that needs to be read in
    """

    def __len__(self)->int:
        return len(self.data)
    
    # implement transform to the labels here
    def __getitem__(self, idx: Any)->dict:
        """
        Fetch a single item given an index

        Args:
            idx (Any): index of the desired item

        Returns:
            Tuple[torch.Tensor, Tuple[int, int]]: returns a tensor representation of the image and tuple of coordinates describing the nose

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path = os.path.join(self.datapath, self.image_paths[idx])
        image = io.imread(path)                                         # this should be a tensor (https://pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image)
        snout_center = self.snout_tuples[idx]

        print(type(image))

        sample = {'image':image, 'center':snout_center}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    filepath = Path("train_noses.txt")
    datapath = Path("images")

    transform_pipeline = transforms.Compose([ToTensor()])

    # TODO: change this to conform to the signature of __getitem__()
    ds = SnoutDataset(infopath=filepath, datapath=datapath)
    sample = ds.__getitem__(1)

    show_landmarks(sample['image'], sample['center'])

    # print(f"sample img path {sample[0]} sample coordinates {type(sample[1])}")
