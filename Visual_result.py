import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from unet_noise_model import UNet

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "/mnt/DATA/EE22B013/noise2noise/checkpoints6/unet_best.pth"
IMAGE_SIZE = 512  # match training/test size if needed

# -----------------------------
# Load model
# -----------------------------
model = UNet(n_channels=1, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# -----------------------------
# Normalization helpers (NONE)
# -----------------------------
def normalize(img_array):
    """Convert uint8 → [0,1]"""
    return img_array.astype(np.float32) / 255.0

def denormalize(t):
    """Tensor → numpy in [0,1]"""
    img = t.squeeze().detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    return img

# -----------------------------
# Image loading
# -----------------------------
def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.array(img).astype(np.float32)
    arr = normalize(arr)   # only /255
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor

# -----------------------------
# Visualization
# -----------------------------
def show_results(noisy_tensor_1,noisy_tensor_2, denoised_tensor, gt_tensor=None, title=""):
    noisy_img_1 = denormalize(noisy_tensor_1)
    noisy_img_2 = denormalize(noisy_tensor_2)
    denoised_img = denormalize(denoised_tensor)

    imgs = [noisy_img_1, noisy_img_2, denoised_img]
    titles = ["Noisy 1", "Noisy 2", "Denoised Output"]

    if gt_tensor is not None:
        gt_img = denormalize(gt_tensor)
        imgs.append(gt_img)
        titles.append("Ground Truth")

    plt.figure(figsize=(20, 5))
    for i, (img, t) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(t)
        plt.axis('off')
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Visual test
# -----------------------------
# noisy1_path = "/mnt/DATA/EE22B013/noise2noise/test/L067/noisy1/slice_003_noisy1.png"
noisy1_path = "/mnt/DATA/EE22B013/noise2noise/output_dataset/noisy1/L067_slice_003_noisy1.png"
noisy2_path = "/mnt/DATA/EE22B013/noise2noise/output_dataset/noisy2/L067_slice_003_noisy2.png"
gt_path     = "/mnt/DATA/EE22B013/noise2noise/output_dataset/groundtruth/L067_slice_003_groundtruth.png"

noisy1 = load_image(noisy1_path)
noisy2 = load_image(noisy2_path)
gt = load_image(gt_path)

with torch.no_grad():
    denoised = model(noisy1)

show_results(noisy1, noisy2, denoised, gt, title="Noise2Noise Denoising Result")
