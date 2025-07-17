import torch
import torch.nn as nn
from pathlib import Path
import os

from mAtt.optimizer import MixOptimizer
from sklearn.metrics import roc_auc_score
import numpy as np

# Allowlist the global
torch.serialization.add_safe_globals(['mAtt_bci'])

def trainNetwork(net, trainloader, validloader, testloader, model_path, iterations=500, lr=5e-4, wd=0.0, repeat=None, sub=None, epochs=None, device=None):
   
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    CE = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
    optimizer = MixOptimizer(optimizer)
    
    best_val_loss = float('inf')
    best_model_path = None
    
    for ite in range(iterations):
        # ----- Training Phase -----
        net.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = net(xb)
            loss = CE(outputs, yb)
            loss.backward()
            optimizer.step()
            
            batch_size = yb.size(0)
            running_train_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            total_train += batch_size
        
        avg_train_loss = running_train_loss / total_train if total_train > 0 else 0
        train_acc = correct_train / total_train if total_train > 0 else 0
        
        # ----- Validation Phase -----
        net.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for xb, yb in validloader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = net(xb)
                loss = CE(outputs, yb)
                batch_size = yb.size(0)
                running_val_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct_val += (preds == yb).sum().item()
                total_val += batch_size
        
        avg_val_loss = running_val_loss / total_val if total_val > 0 else 0
        val_acc = correct_val / total_val if total_val > 0 else 0
        print(f"\nIteration {ite + 1}/{iterations} Train Loss: {avg_train_loss:.4f}   Val Loss: {avg_val_loss:.4f} Train Acc: {train_acc:.4f}   Val Acc: {val_acc:.4f}")
        # print(f"\nIteration {ite + 1}/{iterations}")
        # print(f"Train Loss: {avg_train_loss:.4f}   Val Loss: {avg_val_loss:.4f}")
        # print(f"Train Acc: {train_acc:.4f}   Val Acc: {val_acc:.4f}")
        
        # Save the model if the validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            filename = f"repeat{repeat}_sub{sub}_epochs{epochs}_lr{lr}_wd{wd}.pt"
            save_path = model_dir / filename
            print(f"Saving best model to {save_path}")
            torch.save(net.state_dict(), save_path)
            
            # Evaluate the test set using the current best model
            test_acc = testNetwork(net, testloader, device=device)
            print(f"Test Acc: {test_acc:.4f}")
            best_model_path = save_path

    if best_model_path is not None:
        # Load the best model state into net
        net.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
        net.to(device)
    else:
        print("Warning: No best model was saved during training.")
    
    return net


def testNetwork(net, testloader, device=None):
   
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in testloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = net(xb)
            preds = outputs.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0


def testNetwork_auc(net, testloader, device=None):
   
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for xb, yb in testloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = net(xb)
            probabilities = torch.softmax(outputs, dim=1)
            # Assuming binary classification; uses the probability for class 1
            y_pred.append(probabilities[:, 1].cpu().numpy())
            y_true.append(yb.cpu().numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return roc_auc_score(y_true, y_pred)