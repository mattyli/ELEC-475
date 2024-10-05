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
from model import SnoutNet, EuclideanLoss
from pathlib import Path
from torchinfo import summary
from tqdm import tqdm
import os

# colab dependencies (https://medium.com/swlh/leverage-google-colab-gpu-runtime-for-your-non-notebook-python-project-d13840c932eb)
# from google.colab import drive
# drive.mount('/content/drive/My Drive')

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

# Play around with this
def init_weights(layer: nn.Module):
    """
    Initialize layer weights
    Args:
        layer (nn.Module): Network layer
    """
    if isinstance(layer, nn.Conv2d):
        # nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0.0)
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

# training loop
def train(n_epochs,
          optimizer,
          model,
          loss_fn,
          train_loader,
          validation_loader,
          scheduler,
          device,
          save_file,
          plot_file
          )->None:

    losses_train = []
    losses_validation = []
    for epoch in tqdm(range(n_epochs)):
        print(f"Epoch: {epoch+1}")
        loss_train = 0.0
        
        # train loop
        model.train()
        for batch in train_loader:
            images, centers = batch['image'], batch['center']       # load images and ground truth (labelled centers)
            images = images.to(device=device).float()
            centers = centers.to(device=device).float()                          
            predicted_centers = model(images)                       
            loss = loss_fn(predicted_centers, centers)              

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # validation loop
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for validation_batch in tqdm(validation_loader, leave=False):
                validation_images, validation_centers = validation_batch['image'], validation_batch['center']
                validation_images = validation_images.to(device=device).float()
                validation_centers = validation_centers.to(device=device).float()
                validation_pred = model(validation_images)
                loss = loss_fn(validation_pred, validation_centers)
                validation_loss += loss.item()

        scheduler.step(loss_train)
        epoch_train_loss = loss_train/len(train_loader)
        epoch_validation_loss = validation_loss/ len(validation_loader)
        losses_train += [epoch_train_loss]
        losses_validation += [epoch_validation_loss]
        print(f"{datetime.datetime.now()} Epoch: {epoch+1}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_validation_loss}")
        #   break   # debug
    
    # moved to outside the loop, don't need to redraw image every epoch
    if save_file:
        torch.save(model.state_dict(), save_file)
    if plot_file:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.plot(losses_validation, label='validation')
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
    dropout = DROPOUT_P

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='# of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='output loss plot file (.png)')
    argParser.add_argument('-d', metavar='dropout rate', type=float, help='dropout rate')

    args = argParser.parse_args()

    if args.s != None:
        save_file = args.s
    if args.e != None:
        n_epochs = args.e
    if args.b != None:
        batch_size = args.b
    if args.p != None:
        plot_file = args.p
    if args.d != None and args.d <= 1.0 and args.d >= 0.0:
        dropout = args.d

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\t\tusing device ', device)

    model = SnoutNet(kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, dropout=dropout)
    model.to(device=device)
    model.apply(init_weights)
    summary(model)
    transform_pipeline = transforms.Compose([snoutTransforms.RescaleImage(IMAGE_SIZE), snoutTransforms.ToTensor()])

    print("Fetching training set...")
    train_set = SnoutDataset(label_path="train_noses.txt", image_folder="images/", transform=transform_pipeline)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    print("Fetching validation set...")
    validation_set = SnoutDataset(label_path="test_noses.txt", image_folder="images/", transform=transform_pipeline)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    print("optimizer created...")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    print("scheduler created...")
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    print("loss function created...")
    print('Entering training loop...')

    train(n_epochs=n_epochs,
          optimizer=optimizer,
          model=model,
          loss_fn=loss_fn,
          train_loader=train_loader,
          validation_loader=validation_loader,
          scheduler=scheduler,
          device=device,
          save_file=save_file,
          plot_file=plot_file
          )

if __name__ == '__main__':
    main()