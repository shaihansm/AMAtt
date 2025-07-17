import torch
import torch.utils.data as Data
from scipy import io
from pathlib import Path
from typing import Tuple

def getAllDataloader(subject: int, ratio: int = 8, data_path: str = './data/MAMEM/', bs: int = 64) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
    
    # Set device (uses GPU if available; otherwise, CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Construct the file path using pathlib
    file_path = Path(data_path) / f"U{subject:03d}.mat"
    
    # Load the MATLAB file
    mat_data = io.loadmat(file_path)
    
    # Convert MATLAB arrays into torch tensors with explicit dtypes
    tempdata = torch.tensor(mat_data['x_test'], dtype=torch.float32).unsqueeze(1)
    templabel = torch.tensor(mat_data['y_test'], dtype=torch.long).view(-1)
    
    # Create training, validation, and test splits
    x_train = tempdata[:300]
    y_train = templabel[:300]

    x_valid = tempdata[300:400]
    y_valid = templabel[300:400]

    x_test = tempdata[400:500]
    y_test = templabel[400:500]
    
    # Move tensors to the chosen device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    # Print shapes for debugging
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_valid shape: {x_valid.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
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