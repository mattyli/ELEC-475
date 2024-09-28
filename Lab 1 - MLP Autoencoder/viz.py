import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import autoencoderMLP4Layer

train_transform = transforms.Compose([transforms.ToTensor()])
MNIST = torchvision.datasets.MNIST
IMG_PATH = './imgs'
TRAINED_MODEL = 'MLP.8.pth'

# need to modify this with elements of test.py


def visualize(idx: int):
    mnist = MNIST(root='./data', train=True, download=True, transform=train_transform)
    fig, ax = plt.subplots(1, 2)

    # idx 0 is the actual
    ax[0].imshow(mnist.data[idx], cmap='gray')
    ax[0].set_title(mnist.targets[idx])

    # idx 1 is the reconstruction

    fig.savefig(f'imgs/MNIST index {idx}.png')

def main():
    idx = int(input('Select an index: '))
    if not isinstance(idx, int):
        print('idx must be an int!')
        quit()
    visualize(idx)

if __name__ == '__main__':
    main()
