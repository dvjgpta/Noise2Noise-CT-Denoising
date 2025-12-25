import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR = "/mnt/DATA/EE22B013/noise2noise/train"  # parent folder with L067, L092, etc.
PATCH_SIZE = 128
STRIDE = 64  # overlap of 50% (can tweak)


def extract_patches(img, patch_size=128, stride=64):
    """Extract overlapping patches from an image array."""
    patches = []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches


def make_patches_for_folder(folder_path, patch_size=128, stride=64):
    """Creates patch images for a single folder (e.g., noisy1)."""
    patch_folder = folder_path + "_patches"
    os.makedirs(patch_folder, exist_ok=True)

    img_files = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for fname in tqdm(img_files, desc=f"Processing {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, fname)
        img = np.array(Image.open(img_path).convert("L"))
        patches = extract_patches(img, patch_size, stride)

        # save patches
        base_name = os.path.splitext(fname)[0]
        for idx, patch in enumerate(patches):
            patch_img = Image.fromarray(patch)
            patch_img.save(os.path.join(patch_folder, f"{base_name}_p{idx:03d}.png"))


def main():
    # iterate over subfolders like L067, L092...
    for case_folder in sorted(os.listdir(ROOT_DIR)):
        case_path = os.path.join(ROOT_DIR, case_folder)
        if not os.path.isdir(case_path):
            continue

        print(f"\n📁 Processing case: {case_folder}")
        for sub in ["noisy1", "noisy2"]:
            sub_path = os.path.join(case_path, sub)
            if os.path.exists(sub_path):
                make_patches_for_folder(sub_path, PATCH_SIZE, STRIDE)
            else:
                print(f"⚠️ Skipping missing folder: {sub_path}")


if __name__ == "__main__":
    main()
