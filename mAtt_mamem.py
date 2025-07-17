#!/usr/bin/env python3.9
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import os

from utils.functions import trainNetwork, testNetwork
from mAtt.mAtt import mAtt_mamem
from utils.GetMamem import getAllDataloader

def main():
    # Argument parser for command line options
    parser = argparse.ArgumentParser(
        description="Train mAtt_mamem model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--repeat', type=int, default=1, help='Repetition number for training model')
    parser.add_argument('--sub', type=int, default=1, help='Subject number you want to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--iterations', type=int, default=500, help='Number of training iterations')
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs for splitting EEG signals')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--model_path', type=str, default='./checkpoint/mamem/', help='Folder path for saving the model')
    parser.add_argument('--data_path', type=str, default='./data/MAMEM/', help='Data path')
    args = parser.parse_args()
    
    args_dict = vars(args)
    print(f'Subject: {args_dict["sub"]}')
    
    # Set device: use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataloaders for training, validation, and testing
    trainloader, validloader, testloader = getAllDataloader(
        subject=args_dict['sub'],
        ratio=8,
        data_path=args_dict['data_path'],
        bs=args_dict['bs']
    )
    
    # Instantiate the network and move it to the appropriate device
    net = mAtt_mamem(args_dict['epochs']).to(device)
    
    # Remove keys not needed by the trainNetwork() function
    train_kwargs = args_dict.copy()
    train_kwargs.pop('bs')
    train_kwargs.pop('data_path')
    
    # Train the network
    trainNetwork(net, trainloader, validloader, testloader, **train_kwargs)
    
    # Construct the model filename and directory using pathlib
    model_filename = f'repeat{args_dict["repeat"]}_sub{args_dict["sub"]}_epochs{args_dict["epochs"]}_lr{args_dict["lr"]}_wd{args_dict["wd"]}.pt'
    model_dir = Path(args_dict["model_path"])
    model_path = model_dir / model_filename
    
    # Ensure the model directory exists
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the saved checkpoint (mapping to the proper device)
    if model_path.is_file():
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No best model was saved during training.")
    
    # Test the network and print accuracy
    acc = testNetwork(net, testloader, device=device)
    print(f'Test Accuracy: {acc * 100:.2f}%')

if __name__ == '__main__':
    main()




