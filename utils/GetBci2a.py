import torch
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
from pathlib import Path
from typing import Tuple

def split_train_valid_set(x_train: torch.Tensor, y_train: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
   
    # Sort the training data
    s = y_train.argsort()
    x_train = x_train[s]
    y_train = y_train[s]

    # Calculate the segment length for each of the 4 classes
    cL = len(x_train) // 4

    class1_x = x_train[0 * cL : 1 * cL]
    class2_x = x_train[1 * cL : 2 * cL]
    class3_x = x_train[2 * cL : 3 * cL]
    class4_x = x_train[3 * cL : 4 * cL]

    class1_y = y_train[0 * cL : 1 * cL]
    class2_y = y_train[1 * cL : 2 * cL]
    class3_y = y_train[2 * cL : 3 * cL]
    class4_y = y_train[3 * cL : 4 * cL]

    # Determine the validation length per class
    vL = len(class1_x) // ratio

    # Create training and validation splits
    x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
    y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

    x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
    y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

    return x_train, y_train, x_valid, y_valid


def getAllDataloader(subject: int, ratio: int, data_path: str, bs: int) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
    
    # Convert data_path to a Path object for modern path manipulation.
    data_path = Path(data_path)
    
    # Construct file paths
    train_path = data_path / f'BCIC_S{subject:02d}_T.mat'
    test_path  = data_path / f'BCIC_S{subject:02d}_E.mat'

    # Load data using scipy.io.loadmat
    train = loadmat(train_path)
    test = loadmat(test_path)

    # Convert to torch tensors with explicit data types
    x_train = torch.tensor(train['x_train'], dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train['y_train'], dtype=torch.long).view(-1)
    x_test = torch.tensor(test['x_test'], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test['y_test'], dtype=torch.long).view(-1)

    # Split the training set to form a separate validation set
    x_train, y_train, x_valid, y_valid = split_train_valid_set(x_train, y_train, ratio)

    # Use CPU for dataset creation; move tensors to device after DataLoader creation if needed.
    device = torch.device("cpu")
    # Crop each tensor to use the segment from index 124 to 562 along the last dimension.
    x_train = x_train[:, :, :, 124:562].to(device)
    y_train = y_train.to(device)
    x_valid = x_valid[:, :, :, 124:562].to(device)
    y_valid = y_valid.to(device)
    x_test  = x_test[:, :, :, 124:562].to(device)
    y_test  = y_test.to(device)

    # Print tensor shapes for debugging
    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Create TensorDatasets
    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    test_dataset  = Data.TensorDataset(x_test, y_test)

    # Create DataLoaders
    trainloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    testloader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return trainloader, validloader, testloader