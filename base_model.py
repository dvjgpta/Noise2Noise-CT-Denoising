#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

# assumes your UNet uses signature: UNet(n_channels=..., n_classes=...)
from unet_model2 import UNet
from utils import *          # keep your helper functions here

from datetime import datetime
import time
import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""
        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lehtinen et al., 2018)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if getattr(self.p, 'noise_type', '') == 'mc':
            self.is_mc = True
            # UNet in your repo expects n_channels arg name
            self.model = UNet(n_channels=9, n_classes=1)
        else:
            self.is_mc = False
            self.model = UNet(n_channels=1, n_classes=1)

        # device selection
        self.device = torch.device('cuda' if (
            torch.cuda.is_available() and getattr(self.p, 'cuda', False)) else 'cpu')

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=tuple(self.p.adam[:2]),
                              eps=self.p.adam[2])

            # Learning rate adjustment: patience must be an int
            patience_val = int(self.p.nb_epochs /
                               4) if self.p.nb_epochs >= 4 else 1
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                                                            patience=patience_val,
                                                            factor=0.5,
                                                            verbose=True)

            # Loss function
            if getattr(self.p, 'loss', 'l2') == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif getattr(self.p, 'loss', 'l2') == 'l2':
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.L1Loss()

        # Move model/loss to device
        self.use_cuda = (self.device.type == 'cuda')
        self.model = self.model.to(self.device)
        if self.trainable:
            self.loss = self.loss.to(self.device)

    def _print_params(self):
        """Formats parameters to print when training."""
        print('Training parameters: ')
        # reflect device back into params for printing if needed
        self.p.cuda = (self.device.type == 'cuda')
        param_dict = vars(self.p)
        def pretty(x): return x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v))
              for k, v in param_dict.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if getattr(self.p, 'clean_targets', False):
                ckpt_dir_name = f"{self.p.noise_type}-clean-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            else:
                ckpt_dir_name = f"{self.p.noise_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            if getattr(self.p, 'ckpt_overwrite', False):
                if getattr(self.p, 'clean_targets', False):
                    ckpt_dir_name = f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Save checkpoint dictionary
        if getattr(self.p, 'ckpt_overwrite', False):
            fname_unet = os.path.join(
                self.ckpt_dir, f'n2n-{self.p.noise_type}.pt')
        else:
            # Stats should have valid_loss for current epoch
            valid_loss = stats.get('valid_loss', [None] * (epoch+1))[epoch]
            if valid_loss is None:
                # fallback to timestamp name
                fname_unet = os.path.join(
                    self.ckpt_dir, f'n2n-epoch{epoch+1}.pt')
            else:
                fname_unet = os.path.join(
                    self.ckpt_dir, f'n2n-epoch{epoch+1}-{valid_loss:1.5f}.pt')

        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = os.path.join(self.ckpt_dir, 'n2n-stats.json')
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""
        print('Loading checkpoint from: {}'.format(ckpt_fname))
        # load_state_dict handles map_location automatically if using CPU-only
        map_loc = None if self.use_cuda else torch.device('cpu')
        self.model.load_state_dict(torch.load(
            ckpt_fname, map_location=map_loc))

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves stats after each epoch."""
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='', flush=True)
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        if self.trainable:
            self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if getattr(self.p, 'plot_stats', False):
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss',
                           stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR',
                           stats['valid_psnr'], 'PSNR (dB)')

    def test(self, test_loader, show):
        """Evaluates denoiser on test set and saves montages."""

        self.model.eval()

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        denoised_dir = os.path.dirname(getattr(self.p, 'data', ''))
        if denoised_dir == '':
            denoised_dir = os.getcwd()
        save_path = os.path.join(denoised_dir, 'denoised')
        os.makedirs(save_path, exist_ok=True)

        for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first <show> batches
            if show == 0 or batch_idx >= show:
                break

            source = source.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                denoised_img = self.model(source)

            # store CPU tensors for saving/display
            source_imgs.append(source.cpu())
            denoised_imgs.append(denoised_img.cpu())
            clean_imgs.append(target.cpu())

        # squeeze tensors (assumes batch dimension)
        source_imgs = [t.squeeze(0) if t.dim() == 4 and t.size(
            0) == 1 else t for t in source_imgs]
        denoised_imgs = [t.squeeze(0) if t.dim() == 4 and t.size(
            0) == 1 else t for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) if t.dim() == 4 and t.size(
            0) == 1 else t for t in clean_imgs]

        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        dataset = test_loader.dataset
        # Try to get filenames from dataset; fallback to index
        filenames = getattr(dataset, 'noisy1_files',
                            None) or getattr(dataset, 'imgs', None)
        for i in range(len(source_imgs)):
            if filenames and i < len(filenames):
                img_name = filenames[i]
            else:
                img_name = f'image_{i}'
            create_montage(img_name, self.p.noise_type, save_path,
                           source_imgs[i], denoised_imgs[i], clean_imgs[i], show)

    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.eval()

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source, target) in enumerate(valid_loader):
            source = source.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                source_denoised = self.model(source)

            # Update loss
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            # Compute PSNR per sample in batch (handle last smaller batch)
            source_denoised_cpu = source_denoised.cpu()
            target_cpu = target.cpu()
            batch_size_actual = source_denoised_cpu.size(0)
            for i in range(batch_size_actual):
                psnr_val = psnr(source_denoised_cpu[i], target_cpu[i]).item()
                psnr_meter.update(psnr_val)

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        # report_interval must divide total number of batches OR you can warn instead of assert
        if self.p.report_interval <= 0:
            raise ValueError("report_interval must be > 0")
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'noise_type': getattr(self.p, 'noise_type', None),
                 'noise_param': getattr(self.p, 'noise_param', None),
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches,
                             self.p.report_interval, loss_meter.val)

                source = source.to(self.device)
                target = target.to(self.device)

                # Denoise image
                source_denoised = self.model(source)

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                if self.trainable:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches,
                                   loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg,
                               epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""
        super(HDRLoss, self).__init__()
        self._eps = eps

    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""
        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
