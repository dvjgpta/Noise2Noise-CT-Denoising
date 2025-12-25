import os
import torch
from tqdm import tqdm
from dataset import Noise2NoiseCTDataset, get_ct_dataloader
from unet_model import UNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512
BATCH_SIZE = 1  # typically 1 for testing to save memory

TEST_ROOT = "/mnt/DATA/EE22B013/noise2noise/test"  # parent folder containing L043, L058
CHECKPOINT = "/mnt/DATA/EE22B013/noise2noise/checkpoints6/unet_best.pth"  # last trained model

# -----------------------------
# Load model
# -----------------------------
model = UNet(n_channels=1, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# -----------------------------
# Metric helpers
# -----------------------------
def compute_metrics(pred, target):
    pred_np = (pred.squeeze().cpu().numpy() * 0.5) + 0.5
    target_np = (target.squeeze().cpu().numpy() * 0.5) + 0.5
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    return psnr_val, ssim_val

# -----------------------------
# Test loop
# -----------------------------
all_folder_metrics = []

for folder_name in os.listdir(TEST_ROOT):
    folder_path = os.path.join(TEST_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"\nTesting on dataset: {folder_name}")

    folder_metrics = []

    for noisy_name in ["noisy1", "noisy2"]:
        noisy_dir = os.path.join(folder_path, noisy_name)
        gt_dir = os.path.join(folder_path, "groundtruth")

        test_loader = get_ct_dataloader(noisy_dir, gt_dir,
                                        batch_size=BATCH_SIZE,
                                        image_size=IMAGE_SIZE,
                                        num_workers=8,
                                        shuffle=False)

        psnr_list, ssim_list = [], []

        for noisy_img, gt_img in tqdm(test_loader):
            noisy_img = noisy_img.to(DEVICE)
            gt_img = gt_img.to(DEVICE)

            with torch.no_grad():
                output = model(noisy_img)

            p, s = compute_metrics(output, gt_img)
            psnr_list.append(p)
            ssim_list.append(s)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        folder_metrics.append((avg_psnr, avg_ssim))
        print(f"{noisy_name} → GT: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")

    # Average metrics for this folder (both noisy1 and noisy2)
    folder_avg_psnr = sum([m[0] for m in folder_metrics]) / len(folder_metrics)
    folder_avg_ssim = sum([m[1] for m in folder_metrics]) / len(folder_metrics)
    all_folder_metrics.append((folder_avg_psnr, folder_avg_ssim))

    print(f"Folder {folder_name} average: PSNR={folder_avg_psnr:.4f}, SSIM={folder_avg_ssim:.4f}")

# -----------------------------
# Overall metrics across all folders
# -----------------------------
total_avg_psnr = sum([m[0] for m in all_folder_metrics]) / len(all_folder_metrics)
total_avg_ssim = sum([m[1] for m in all_folder_metrics]) / len(all_folder_metrics)

print("\n=====================")
print(f"Overall Test Metrics: PSNR={total_avg_psnr:.4f}, SSIM={total_avg_ssim:.4f}")
print("=====================")
