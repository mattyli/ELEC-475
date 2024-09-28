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
from typing import Union

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample: dict):
        pass

# TODO: should the resize tuple be hardcoded into this? <-- not good for modularity, but can't pass as a param if being used with compose?
# TODO: is sample passed implicitly?
# TODO: this example pretty much does what we want, just need to change it from multiple landmarks to one (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
class ResizeImage(object):
    """
    Class to resize an image given a tuple of dimensions to resize to.
    Will resize image and alter coordinates accordingly
    """

    def __init__(self, output_size: Union[int, tuple[int, int]]):
        assert isinstance(output_size, (tuple, int))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2    # make sure that the input is actually a 2-ple
            self.output_size = output_size
            
    # TODO: verify whether the coordinate order in the text file is x or y
    def __call__(self, sample: dict):       # expects a dictionary of {image, centerpoint}
        pass
