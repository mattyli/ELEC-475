import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import snoutTransforms
import matplotlib.pyplot as plt
import datetime
import argparse
from torch.utils.data import DataLoader
from SnoutDataset import SnoutDataset
from model import SnoutNet
from pathlib import Path
from torchinfo import summary


# global vars and hyperparameters
SAVE_FILE = "weights.pth"
N_EPOCHS = 50
BATCH_SIZE = 256
IMAGE_SIZE = (227, 227)
LOSS_PLOT = f"SnoutNet Loss b: {BATCH_SIZE}, epochs: {N_EPOCHS}"
KERNEL_SIZE = 3                                                     # kernel size for maxpool and conv
STRIDE = 2                                                          # stride for maxpool and conv
PADDING = 1                                                         # padding for maxpool and conv
DROPOUT_P = 0.0                                                     # dropout probability
IMAGE_FOLDER = Path("images/")
TRAIN_PATH = Path("train_noses.txt")
TEST_PATH = Path("test_noses.txt")

def init_weights(layer: nn.Module):
    """
    Initialize layer weights
    Args:
        layer (nn.Module): Network layer
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        nn.init.constant_(layer.bias, 0)
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

def main():

    global bottleneck_size, save_file, n_epochs, batch_size

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')

    args = argParser.parse_args()

    if args.s != None:
        save_file = args.s
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\t\tusing device ', device)

    model = SnoutNet(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, dropout=DROPOUT_P)
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)

    # TODO: 
    train_transform = transforms.Compose([snoutTransforms.RescaleImage(IMAGE_SIZE), snoutTransforms.ToTensor()])
    test_transform = train_transform

    train_set = SnoutDataset()

    # train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    # # test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
