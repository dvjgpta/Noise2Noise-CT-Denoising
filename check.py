# compute basic stats over a small sample or whole dataset
from PIL import Image
import numpy as np, os, random

from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio

a = imageio.imread("/mnt/DATA/EE22B013/noise2noise/output_dataset/noisy1/L067_slice_001_noisy1.png")
b = imageio.imread("/mnt/DATA/EE22B013/noise2noise/output_dataset/noisy2/L067_slice_001_noisy2.png")
gt = imageio.imread("/mnt/DATA/EE22B013/noise2noise/output_dataset/groundtruth/L067_slice_001_groundtruth.png")

print(psnr(gt, a))
print(psnr(gt, b))
print(psnr(a, b))
