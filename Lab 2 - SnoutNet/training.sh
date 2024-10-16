#!/bin/bash
cd drive/MyDrive/School_2024-2025/ELEC475-ComputerVision/Lab_2-SnoutNet
pip install torchinfo
unzip images.zip -d "/content/data"

echo "Training control model..."
python train.py -fp "/content/data" -s e.50.b.48.control.pth -e 50 -b 48 -p e50_b48_control.png

echo "Training flip only..."
python train.py -fp "/content/data" -s e.50.b.48.flip_only.pth -e 50 -b 48 -p e50_b48_flip_only.png --hflip --vflip

echo "Training RCJ only..."
python train.py -fp "/content/data" -s e.50.b.48.RCJ_only.pth -e 50 -b 48 -p e50_b48_RCJ_only.png --rcj

echo "Training all..."
python train.py -fp "/content/data" -s e.50.b.48.all_transform.pth -e 50 -b 48 -p e50_b48_all_transform.png --rcj --hflip --vflip