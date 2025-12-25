import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from unet_noise_model import UNet
from dataset import get_ct_dataloader ,get_test_loader
from torch.optim import lr_scheduler
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
LR = 1e-4
BATCH_SIZE = 8
IMAGE_SIZE = 256

# IMPORTANT: must match dataset.py Normalize()
# MEAN = 0.1725
# STD = 0.2579

ROOT_DIR = "/mnt/DATA/EE22B013/noise2noise/train/merged_train"
TEST_DIR = "/mnt/DATA/EE22B013/noise2noise/test/L067" 

# --------------------------------------------------------
# Correct PSNR computation for standardized images
# --------------------------------------------------------
# def compute_metrics(pred, target):
#     pred_np = pred.squeeze().cpu().numpy()
#     target_np = target.squeeze().cpu().numpy()

#     # invert normalization
#     pred_np = (pred_np * STD) + MEAN
#     target_np = (target_np * STD) + MEAN

#     pred_np = np.clip(pred_np, 0, 1)
#     target_np = np.clip(target_np, 0, 1)

#     return psnr(target_np, pred_np, data_range=1.0)
def compute_metrics(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    return psnr(target_np, pred_np, data_range=1.0)


import math

def rampup(epoch, rampup_length=10):
    if epoch < rampup_length:
        p = 1.0 - epoch / rampup_length
        return math.exp(-p * p * 5.0)
    return 1.0

def rampdown(epoch, num_epochs, rampdown_length=30):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0

def get_lr(epoch, lr_max, num_epochs):
    return lr_max * rampup(epoch) * rampdown(epoch, num_epochs, 30)

def train():
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    torch.backends.cudnn.benchmark = True

    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    noisy1_dir = os.path.join(ROOT_DIR, "noisy1")
    noisy2_dir = os.path.join(ROOT_DIR, "groundtruth")

    train_loader = get_ct_dataloader(
        noisy1_dir, noisy2_dir,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=16,
        shuffle=True
    )
    test_loader = get_test_loader(
        TEST_DIR,
        batch_size=1,
        image_size=IMAGE_SIZE,
        num_workers=8
    )

    best_loss = float("inf")
    os.makedirs("checkpoints6", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        lr = LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR    

        for noisy_input, noisy_target in loop:
            noisy_input, noisy_target = noisy_input.to(DEVICE), noisy_target.to(DEVICE)

            optimizer.zero_grad()
            output = model(noisy_input)

            loss = criterion(output, noisy_target)
            loss.backward()

            # Prevent catastrophic gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if torch.isnan(loss):
                print("NaN detected — skipping batch")
                continue

            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # STEP LR SCHEDULER AFTER EACH EPOCH (correct)
        

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f}, Using LR = {lr:.6f}")

        # ------------------------------------------------------
        # Compute PSNR every 10 epochs
        # ------------------------------------------------------
        if (epoch + 1) % 5 == 0:
            model.eval()
            total_psnr = 0.0
            count = 0
            with torch.no_grad():
                for noisy_input, clean_target in test_loader:
                    noisy_input = noisy_input.to(DEVICE)
                    clean_target = clean_target.to(DEVICE)

                    output = model(noisy_input)

                    total_psnr += compute_metrics(output, clean_target)
                    count += 1

            avg_psnr = total_psnr / count
            print(f"Epoch [{epoch+1}] - Average PSNR: {avg_psnr:.4f} dB")

        # save checkpoint every 10 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"checkpoints6/unet_epoch_{epoch+1}.pth")

        # save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints6/unet_best.pth")
            print(f"New BEST model saved at epoch {epoch+1} (loss = {best_loss:.6f})")

    torch.save(model.state_dict(), "checkpoints6/unet_final.pth")
    print("Training complete — final model saved")


if __name__ == "__main__":
    train()
