"""
This module contains overridden transforms to be
used with the images and target coordinates

Each callable class must override the __call__() method.
Each __call__() method in the defined classes should have the same argument which is sample

***    Note    ***
If a class is expecting more than an image as the input, then need to specify an
init method to accept the other arguments

"""

import numpy as np
import torch
from typing import Union, Dict
from skimage import transform

# Taken from PyTorch tutorial
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample: Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:
        image, center = sample['image'], sample['center']
        assert isinstance(image, np.ndarray)                # pre-caution
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'center': torch.from_numpy(center)}

# Taken from PyTorch tutorial, modified to fit this task
# TODO: this example pretty much does what we want, just need to change it from multiple landmarks to one (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
class RescaleImage(object):
    """
    Class to resize an image given a tuple of dimensions to resize to.
    Will resize image and alter coordinates accordingly
    """

    def __init__(self, output_size: Union[int, tuple[int, int]])->None:
        assert isinstance(output_size, (tuple, int))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2    # make sure that the input is actually a 2-ple
            self.output_size = output_size

    # TODO: verify whether the coordinate order in the text file is x or y
    def __call__(self, sample: Dict[str, np.ndarray])->Dict[str, torch.Tensor]:       # expects a dictionary of {image, centerpoint}
        image, center = sample['image'], sample['center']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        center = center * [new_w / w, new_h / h]

        return {'image': img, 'center': center}

