import os
import shutil

# -----------------------------
# CONFIGURATION
# -----------------------------
ROOT_DIR = "/mnt/DATA/EE22B013/noise2noise/train"  # path containing L001, L002, etc.
MERGED_DIR = "/mnt/DATA/EE22B013/noise2noise/merged_train"  # output folder

# Create merged output structure
for sub in ["noisy1", "noisy2", "groundtruth"]:
    os.makedirs(os.path.join(MERGED_DIR, sub), exist_ok=True)

# -----------------------------
# MERGE LOOP
# -----------------------------
folders = sorted([f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))])
print(f"Found {len(folders)} folders to merge:", folders)

for folder in folders:
    folder_path = os.path.join(ROOT_DIR, folder)
    print(f"Processing folder: {folder}")

    for subfolder in ["noisy1", "noisy2", "groundtruth"]:
        src_dir = os.path.join(folder_path, subfolder)
        dst_dir = os.path.join(MERGED_DIR, subfolder)

        if not os.path.exists(src_dir):
            print(f"⚠️  Skipping {folder}/{subfolder} — not found.")
            continue

        for filename in os.listdir(src_dir):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
                continue

            src_path = os.path.join(src_dir, filename)

            # Make filename unique by prefixing folder name
            name, ext = os.path.splitext(filename)
            new_name = f"{folder}_{name}{ext}"
            dst_path = os.path.join(dst_dir, new_name)

            # Copy image
            shutil.copy2(src_path, dst_path)

print("\n✅ All folders merged successfully!")
print(f"Merged dataset structure saved in: {MERGED_DIR}")
