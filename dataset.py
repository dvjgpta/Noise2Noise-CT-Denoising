import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import natsort



class Noise2NoiseCTDataset(Dataset):
    def __init__(self, noisy1_dir, noisy2_dir, transform=None):
        """
        Custom dataset for Noise2Noise training with grayscale CT images.

        Args:
            noisy1_dir (str): Path to folder with input noisy images.
            noisy2_dir (str): Path to folder with target noisy images.
            transform (callable, optional): Transform to apply to both images.
        """
        self.noisy1_dir = noisy1_dir
        self.noisy2_dir = noisy2_dir
        self.noisy1_files = natsort.natsorted(os.listdir(noisy1_dir))
        self.noisy2_files = natsort.natsorted(os.listdir(noisy2_dir))
        self.transform = transform

        assert len(self.noisy1_files) == len(self.noisy2_files), \
            f"Mismatch: {len(self.noisy1_files)} vs {len(self.noisy2_files)} images"
        print("\nChecking first 20 filename pairs:")
        for a, b in zip(self.noisy1_files[:20], self.noisy2_files[:20]):
            print(a, "  |  ", b)
        print("--------------------------------------------------")

    def __len__(self):
        return len(self.noisy1_files)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.noisy1_dir, self.noisy1_files[idx])
        img2_path = os.path.join(self.noisy2_dir, self.noisy2_files[idx])

        # Convert to grayscale ('L' mode)
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


def get_ct_dataloader(noisy1_dir, noisy2_dir, batch_size=8, image_size=256, num_workers=8, shuffle=True):
    """
    Builds a DataLoader for CT image Noise2Noise training.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # Output shape: [1, H, W]
        #transforms.Normalize([0.1725], [0.2579])
    ])

    dataset = Noise2NoiseCTDataset(noisy1_dir, noisy2_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_test_loader(test_dir, batch_size=1, image_size=512, num_workers=8):
    """
    Builds DataLoader for testing.
    Expects folder structure:
        test_dir/
            noisy1/
            groundtruth/
    """
    noisy1_dir = os.path.join(test_dir, "noisy1")
    groundtruth_dir = os.path.join(test_dir, "groundtruth")
    
    loader = get_ct_dataloader(noisy1_dir, groundtruth_dir,
                               batch_size=batch_size,
                               image_size=image_size,
                               num_workers=num_workers,
                               shuffle=False)
    return loader


# # Example usage:
# if __name__ == "__main__":
#     noisy1_path = "data/ct_noisy1"
#     noisy2_path = "data/ct_noisy2"

#     loader = get_ct_dataloader(
#         noisy1_path, noisy2_path, batch_size=4, image_size=256)

#     for i, (noisy_input, noisy_target) in enumerate(loader):
#         print(
#             f"Batch {i}: input shape = {noisy_input.shape}, target shape = {noisy_target.shape}")
#         if i == 1:
#             break
