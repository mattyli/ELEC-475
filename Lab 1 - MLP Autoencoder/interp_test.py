
#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model_interp import autoencoderMLP4Layer
import random

def main():

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')
    argParser.add_argument('-n', metavar='interpolation amount', type=int, help='int [32]')     # added param to specify the number of interpolations done

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 8
    if args.z != None:
        bottleneck_size = args.z
    n_interpolations = 8
    if args.n != None:
        n_interpolations = args.n
    # end arguements

    # device is the processing unit available
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    current_set = test_set

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)

    
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()
    # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
    # torch.Tensor.view seems to behave very similarly to np.reshape()
    # negative index exits the program
    idx = 0
    while idx >= 0:
        idx = int(input("Enter index > "))
        idx_2 = int(input("Enter second index > "))
        if 0 <= idx <= current_set.data.size()[0] and 0 <= idx_2 <= current_set.data.size()[0]:
            print('label = ', current_set.targets[idx].item())
            img = current_set.data[idx]
            img2 = current_set.data[idx_2]    # arbitrarily picking a random index

            img = img.type(torch.float32)                           # what do these 3 lines do?
            img = (img - torch.min(img)) / torch.max(img)           # normalizing the image to the range of 0-255
            img = img.to(device=device)                             # write the image to memory?

            img2 = img2.type(torch.float32)                         # what do these 3 lines do?
            img2 = (img2 - torch.min(img2)) / torch.max(img2)
            img2 = img2.to(device=device)                           # write the image to memory?

            # must reshape before passing to model
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
            img2 = img2.view(1, img2.shape[0]*img2.shape[1]).type(torch.FloatTensor)

            # Add noise with torch.rand(size=int or Collection[int])
            noise_filter = torch.rand(size=img.size())          # because img is a torch.tensor
            noisy_img = img + noise_filter                      # actually adding the noise to the image (this is just matrix addition  (elementwise addition))
            noisy_img = noisy_img.to(device=device)  
            noisy_img = noisy_img.view(1, 28*28).type(torch.FloatTensor)    # called view because the underlying data in memory is the same

            print(f"Noisy Image on device: {noisy_img.get_device()}")
            print(f'NOISY SHAPE: {noisy_img.size()} \n IMG SHAPE: {img.size()}')

            with torch.no_grad():
                reconstruction = model(img)
                output = model(noisy_img)
                b1 = model.encode(img)      # model.encode expects a tensor of (1,784)
                b2 = model.encode(img2)

                interpolations = torch.zeros(n_interpolations, b1.shape[1]) # tensor of tensors

                for terp in range(n_interpolations):
                    weight = terp/ (n_interpolations-1)
                    interpolations[terp] = (1-weight) * b1 + weight * b2    # kind of like turning a dial, slowly ramping up the amount the bottle neck is composed of one and not the other
                
                interpolated_imgs = model.decode(interpolations)            # decode requires a tensor input                                               # model.decode should return a tensor of (1, 784)
                interpolated_imgs = [tensor.view(28,28).type(torch.FloatTensor) for tensor in interpolated_imgs]    # list of 28x28 tensors
          
            # reshaping the tensors into a shape that can be plotted
            output = output.view(28, 28).type(torch.FloatTensor)
            img = img.view(28, 28).type(torch.FloatTensor)
            img2 = img2.view(28, 28).type(torch.FloatTensor)
            noisy_img = noisy_img.view(28, 28).type(torch.FloatTensor)
            reconstruction = reconstruction.view(28, 28).type(torch.FloatTensor)
            
            # VISUALIZATIONS
            # simple encode, decode
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(img, cmap='grey')
            ax[1].imshow(reconstruction, cmap='grey')
            fig.suptitle("Step 4: Encode/Decode")
            fig.show()

            # image denoising
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(img, cmap='grey')
            ax[1].imshow(noisy_img, cmap='grey')
            ax[2].imshow(output, cmap='grey')
            fig.suptitle(f"Step 5: Image Denoising")
            fig.show()

            # TODO: update this to take n + 2 images where n is the number of interpolations
            fig, ax = plt.subplots(1,n_interpolations+2)
            ax[0].imshow(img, cmap='gray')
            for i, picture in enumerate(interpolated_imgs):
                ax[i+1].imshow(picture, cmap='gray')
            ax[-1].imshow(img2, cmap='gray')
            fig.suptitle(f"Step 6: Bottleneck interpolation n = {n_interpolations}")
            fig.show()
            
            
            

            

###################################################################

if __name__ == '__main__':
    main()



