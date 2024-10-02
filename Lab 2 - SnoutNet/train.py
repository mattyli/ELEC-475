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

# Play around with this
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

# training loop
def train(n_epochs,
          optimizer,
          model,
          loss_fn,
          train_loader,
          scheduler,
          device,
          save_file,
          plot_file,
          validation_loader=None
          )->None:
    print("training model...")

    losses_train = []
    model.train()
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}")
        loss_train = 0.0

        for batch in train_loader:
            images, centers = batch['image'], batch['center']       # load images and ground truth (labelled centers)
            images = images.to(device=device)                       
            predicted_centers = model(images)                       
            loss = loss_fn(predicted_centers, centers)              

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]
        print(f"{datetime.datetime.now()} Epoch: {epoch+1}, Training Loss: {loss_train/len(train_loader)}")
    
    # moved to outside the loop, don't need to redraw image every epoch
    if save_file:
        torch.save(model.state_dict(), save_file)
    if plot_file:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        print('saving ', plot_file)
        plt.savefig(plot_file)

def main():

    save_file = "weights.pth"   # default
    plot_file = "loss.png"
    n_epochs = N_EPOCHS
    batch_size = BATCH_SIZE

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
    transform_pipeline = transforms.Compose([snoutTransforms.RescaleImage(IMAGE_SIZE), snoutTransforms.ToTensor()])

    train_set = SnoutDataset(label_path="train_noses.txt", image_folder="images/", transform=transform_pipeline)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train(n_epochs=n_epochs,
          optimizer=optimizer,
          model=model,
          loss_fn=loss_fn,
          train_loader=train_loader,
          scheduler=scheduler,
          device=device,
          save_file=save_file,
          plot_file=plot_file
          )


if __name__ == 'main':
    main()