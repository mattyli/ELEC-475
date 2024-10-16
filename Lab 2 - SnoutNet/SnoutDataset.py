import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from snoutTransforms import ToTensor, RescaleImage, RandomFlip, RandomColorJitter
from typing import Tuple, Any, Optional, Union, List
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

IMAGE_SIZE = (227, 227)

def show_batch(batch: dict):
    batch_images, batch_center = batch['image'], batch['center']
    batch_size = len(batch_images)
    im_size = batch_images.size(2)
    grid_border_size = 2
   
    grid = utils.make_grid(batch_images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        plt.scatter(batch_center[i, 0].numpy() + i * im_size + (i+1) * grid_border_size, 
                    batch_center[i, 1].numpy() + grid_border_size,
                    s=10, marker='*', c='r'
                    )

class SnoutDataset(Dataset):
    def __init__(self, dir: Union[str, Path], train: bool = True, transform=None): 
        self.transform = transform
        self.set_file = os.path.join(dir, "train_noses.txt") if train else os.path.join(dir, "test_noses.txt")
        self.dir = os.path.join(dir, 'images')

        with open(self.set_file, "r") as file:
            self.data = []

            for line in file:
                img_path, snout_center = line.rstrip().split(",", 1)
                if img_path.endswith(('.jpg', '.jpeg')):
                    snout_center = np.asarray(ast.literal_eval(ast.literal_eval(snout_center)), dtype=float)#.reshape(-1,2)
                    img_path = os.path.join(self.dir, img_path)
                    self.data.append((img_path, snout_center))
    
    def __len__(self)->int:
        return len(self.data)
                
    def __getitem__(self, idx: Any)->dict:
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()
    
        while True:
                try:
                    with Image.open(self.data[idx][0]) as img:               # using PIL.Image.open() and .verify() to check the integrity of images, using skimage to actually read
                        img = img.convert('RGB')
                        img.verify()
                        img = np.array(img)                                 # convert to RGB image no matter the input shape
                        snout_center = self.data[idx][1]
                        sample = {'image': img, 'center': snout_center}

                        if self.transform:
                            sample = self.transform(sample)

                        return sample

                except (IOError, SyntaxError) as e:
                    idx += 1 % self.__len__()                               # increment index to the next image

if __name__ == "__main__":

    transform_pipeline = transforms.Compose([RescaleImage(IMAGE_SIZE), RandomFlip("HORIZONTAL", state=True), RandomFlip("VERTICAL", state=True),
                                             ToTensor(),
                                             RandomColorJitter(brightness=2, contrast=2, saturation=2, state=True)
                                             ])

    dataset = SnoutDataset(dir=Path(r"C:\Users\02mat\OneDrive\Desktop\Year 4 - CMPE\ELEC 475\Lab 2 - SnoutNet"), train=True, transform=transform_pipeline)
    print(dataset.__len__())

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    for i, batch in enumerate(tqdm(dataloader)):

        if i == 8:
            break
        print(f"Batch Image size: {batch['image'].size()} \n Batch Center size: {batch['center'].size()}")

        plt.figure(figsize=(10,10))
        show_batch(batch)
        plt.axis('off')
        plt.ioff()
        plt.title(f"Batch {i} from dataloader")
        plt.savefig(f'report_images/batch_{i}_transforms.png')
    
