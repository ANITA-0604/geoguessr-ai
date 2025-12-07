import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import custom modules
from model import GeoGuessorModel
from geoguessr_dataset import GeoGuessrDataset

# --- Configuration ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
GRID_SIZE_X = 20
GRID_SIZE_Y = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./images" # Parent folder containing 'train' and 'val subfolders

def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_reg, device):
    model.train()
    total_loss = 0
    correct_gx = 0
    correct_gy = 0
    total_samples = 0

    loop = tqdm(loader, desc = "Training")

    for images, targets in loop:
        images = images.to(device)
        targets = targets.to(device)

        # --- 1. Unpack Targets ---
        # Dataset returns one big tensor. We need to slice it.
        # indices: 0 = state, 1 = gx, 2 = gy, 3 = off_x, 4 = off_y
        true_gx = targets[:, 1].long()
        true_gy = targets[:, 2].long()
        true_offset = targets[:, 3:5] * 2 - 1

        # --- 2. Forward Pass ---
        # We will pass true grids for "Teacher Forcing" during training
        outputs = model(images, true_gx = true_gx, true_gy = true_gy, use_teacher_forcing = True)

        # --- 3. Compute Loss ---
        # Classification Loss (CrossEntropy)
        loss_gx = criterion_cls(outputs['gx_logits'], true_gx)
        loss_gy = criterion_cls(outputs['gy_logits'], true_gy)

        # Regression Loss (MSE)
        loss_offset = criterion_reg(outputs['offset'], true_offset)

        # Total Loss (Weighted sum)
        # We weight offset less because it's a refinement step
        loss = loss_gx + loss_gy + (5.0 * loss_offset)

        # --- 4. Backward Pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- 5. Metrics ---
        total_loss += loss.item()

        # Calculate Accuracy
        pred_gx = torch.argmax(outputs['gx_logits'], dim = 1)
        pred_gy = torch.argmax(outputs['gy_logits'], dim = 1)

        correct_gx += (pred_gx == true_gx).sum().item()
        correct_gy += (pred_gy == true_gy).sum().item()
        total_samples += images.size(0)

        loop.set_postfix(loss = loss.item())

    avg_loss = total_loss / len(loader)
    acc_gx = correct_gx / total_samples
    acc_gy = correct_gy / total_samples

    return avg_loss, acc_gx, acc_gy

def validate(model, loader, criterion_cls, criterion_reg, device):
    model.eval()
    total_loss = 0
    correct_gx = 0
    correct_gy = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc = "Validation"):
            images = images.to(device)
            targets = targets.to(device)

            true_gx = targets[:, 1].long()
            true_gy = targets[:, 2].long()
            true_offset = targets[:, 3:5] * 2 - 1

            # No teacher forcing in validation, Model must use its own predictions
            outputs = model(images, use_teacher_forcing = False)

            loss_gx = criterion_cls(outputs['gx_logits'], true_gx)
            loss_gy = criterion_cls(outputs['gy_logits'], true_gy)
            loss_offset = criterion_reg(outputs['offset'], true_offset)

            loss = loss_gx + loss_gy + (5.0 * loss_offset)

            total_loss += loss.item()

            pred_gx = torch.argmax(outputs['gx_logits'], dim = 1)
            pred_gy = torch.argmax(outputs['gy_logits'], dim = 1)

            correct_gx += (pred_gx == true_gx).sum().item()
            correct_gy += (pred_gy == true_gy).sum().item()
            total_samples += images.size(0)

    return total_loss / len(loader), correct_gx / total_samples, correct_gy / total_samples

def main():
    # --- Setup Data ---
    # Assumes structure: images/train/meta.jsonl AND images/val/meta.jsonl
    # train_path = os.path.join(DATA_DIR, json_file = 'train_meta.jsonl')
    # val_path = os.path.join(DATA_DIR, json_file = 'val_meta.jsonl')

    train_ds = GeoGuessrDataset(DATA_DIR, json_file = 'train_meta.jsonl')
    val_ds = GeoGuessrDataset(DATA_DIR, json_file = 'val_meta.jsonl')

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)

    # --- Setup Model ---
    model = GeoGuessorModel(
        num_grid_x = GRID_SIZE_X,
        num_grid_y = GRID_SIZE_Y,
        d_model = 512
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    # CrossEntropy for classification(Lat/Lon Grids)
    criterion_cls = nn.CrossEntropyLoss()
    # MSE for regression(Offset inside the grid)
    criterion_reg = nn.MSELoss()

    # --- Training Loop ---
    best_acc = 0.0

    train_loss, train_acc_gx, train_acc_gy = train_one_epoch(
        model, train_loader, optimizer, criterion_cls, criterion_reg, DEVICE
    )

    val_loss, val_acc_gx, val_acc_gy = validation(
        model, val_loader, criterion_cls, criterion_reg, DEVICE
    )

    print(f"Train Loss: {train_loss:.4f} | GridX Acc: {train_acc_gx:.2%} | GridY Acc: {train_acc_gy:.2%}")
    print(f"Val Loss:   {val_loss:.4f} | GridX Acc: {val_acc_gx:.2%} | GridY Acc: {val_acc_gy:.2%}")

    # Save Best Model
    avg_val_acc = (val_acc_gx + val_acc_gy) / 2
    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        torch.save(model.state_dict(), "best_geoguessr_model.pth")
        print("New Best Model Saved!")

if __name__ == "__main__":
    main()
