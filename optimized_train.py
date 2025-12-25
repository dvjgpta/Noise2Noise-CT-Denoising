import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from unet_model import UNet
from dataset import get_ct_dataloader
import os
import time
import matplotlib.pyplot as plt
from pytorch_msssim import ssim  # pip install pytorch-msssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Training Hyperparameters
# =============================
EPOCHS = 150
LR = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 128
DATA_ROOT = "/mnt/DATA/EE22B013/noise2noise/train/merged_train"  # <<-- updated path
NUM_WORKERS = 8
ALPHA = 1  # weight for MSE vs SSIM

# =============================
# Combined MSE + SSIM Loss
# =============================
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
        return total_loss


# =============================
# Training Function
# =============================
def train():
    # --------------------------
    # Load model
    # --------------------------
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    torch.backends.cudnn.benchmark = True
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # --------------------------
    # Loss, optimizer, scheduler
    # --------------------------
    criterion = CombinedLoss(alpha=ALPHA)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25, verbose=True, min_lr=1e-6
    )

    # --------------------------
    # Dataloader (merged dataset)
    # --------------------------
    noisy1_dir = os.path.join(DATA_ROOT, "noisy1_patches")
    noisy2_dir = os.path.join(DATA_ROOT, "noisy2_patches")

    train_loader = get_ct_dataloader(
        noisy1_dir, noisy2_dir,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    print(f"Loaded merged dataset: {len(train_loader.dataset)} image pairs")

    # --------------------------
    # Training Loop
    # --------------------------
    losses = []
    total_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for noisy_input, noisy_target in loop:
            noisy_input, noisy_target = noisy_input.to(DEVICE), noisy_target.to(DEVICE)

            optimizer.zero_grad()
            output = model(noisy_input)
            loss = criterion(output, noisy_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        losses.append(avg_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        eta = (EPOCHS - epoch - 1) * elapsed
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.6f} | "
              f"LR: {current_lr:.2e} | ETA: ~{eta_h}h {eta_m}m")

        # Save model checkpoint periodically
        os.makedirs("checkpoints", exist_ok=True)
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f"checkpoints/unet_epoch{epoch+1}.pth")
            print(f"✅ Saved checkpoint at epoch {epoch+1}")

    total_time = (time.time() - total_start) / 3600
    print(f"\n🚀 Training completed in {total_time:.2f} hours.")

    # --------------------------
    # Plot Training Loss
    # --------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color='royalblue', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve (MSE + SSIM)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train()
