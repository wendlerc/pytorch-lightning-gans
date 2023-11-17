"""
To run this template just do:
python dcgan.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os
import io
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import webdataset as wds

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, n_channels=4, n_feats=128, kernel_size=3):
        super().__init__()
        self.img_shape = img_shape

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, n_feats * self.init_size ** 2))

        self.n_feats = n_feats

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(n_feats),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_feats, n_feats, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(n_feats, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(n_feats, n_feats//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(n_feats//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feats//2, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.n_feats, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, bn=True):
            block = [nn.Conv2d(in_feat, out_feat, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class DCGAN(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, 
                 num_workers: int = 0,
                 n_generator_steps_per_discriminator_step: int = 2,
                 discriminator_grad_clipping: int = 5,
                 log_every_n_steps: int = 100,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_generator_steps_per_discriminator_steps = n_generator_steps_per_discriminator_step
        self.discriminator_grad_clipping = discriminator_grad_clipping
        self.log_every_n_steps = log_every_n_steps

        self.automatic_optimization = False

        # networks
        img_shape = (4, 64, 64)
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.validation_z = torch.randn(8, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        optG, optD = self.optimizers()
        imgs, = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        # train generator
        for _ in range(self.n_generator_steps_per_discriminator_steps):
            optG.zero_grad()
            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:16]
            grid = torchvision.utils.make_grid(sample_imgs)
            if self.global_step % self.log_every_n_steps == 0:
                self.logger.experiment.log({'generated_images': [wandb.Image(grid, caption='Generated Images')]})
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            D_G_z1 = self.discriminator(self.generated_imgs)
            g_loss = self.adversarial_loss(D_G_z1, valid)
            g_loss.backward()
            optG.step()
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        optD.zero_grad()
        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        D_x = self.discriminator(imgs)
        real_loss = self.adversarial_loss(D_x, valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(
            self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        if self.discriminator_grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.discriminator_grad_clipping)
        optD.step()
        tqdm_dict = {'d_loss': d_loss}
        output = OrderedDict({
            'loss': d_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        self.log('epoch', self.current_epoch, on_step=True, on_epoch=True)
        self.log('loss_D', d_loss.item(), on_step=True, on_epoch=True)
        self.log('loss_G', g_loss.item(), on_step=True, on_epoch=True)
        self.log('D(x)', D_x.mean(), on_step=True, on_epoch=True)
        self.log('D(G(z))', D_G_z1.mean(), on_step=True, on_epoch=True)
        self.log('grad_norm_D', torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1), on_step=True, on_epoch=True)
        self.log('grad_norm_G', torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1), on_step=True, on_epoch=True)

        return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        def load_latent(z):
            return torch.load(io.BytesIO(z), map_location='cpu').to(torch.float32)
        
        rescale = torch.nn.Tanh()

        dataset = wds.WebDataset('../../data/latents/{000000..000007}.tar')
        dataset = dataset.rename(image="latent.pt")
        dataset = dataset.map_dict(image=load_latent)
        dataset = dataset.map_dict(image=rescale)
        dataset = dataset.to_tuple("image")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return dataloader

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)

        self.logger.experiment.log({'generated_images': [wandb.Image(grid, caption='Generated Images')]}, step=self.current_epoch)


def main(args: Namespace) -> None:
    # Wandb logging
    wandb_logger = WandbLogger(project=args.wandb_project, 
                               log_model=False, 
                               save_dir=args.checkpoint_path)
    wandb_logger.experiment.config.update(args)
    run_name = wandb_logger.experiment.name

    # Configure the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path, run_name),  # Define the path where checkpoints will be saved
        save_top_k=-1,  # Set to -1 to save all epochs
        verbose=True,  # If you want to see a message for each checkpoint
        monitor='D(x)',  # Quantity to monitor
        mode='min',  # Mode of the monitored quantity
        every_n_train_steps=args.checkpoint_every_n_examples//args.batch_size,
    )

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = DCGAN(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distubuted training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    
    trainer = Trainer(max_epochs=args.max_epochs, accelerator=args.accelerator, logger=wandb_logger, callbacks=[checkpoint_callback])

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="auto", help="auto, dp, ddp, ddp2, ddp_spawn, ddp_cpu, etc.")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=256,
                        help="dimensionality of the latent space")
    parser.add_argument("--wandb_project", type=str, default="dcgan")
    parser.add_argument("--checkpoint_path", type=str, default="../../models/dcgan/")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--n_generator_steps_per_discriminator_step", type=int, default=2)
    parser.add_argument("--discriminator_grad_clipping", type=int, default=5)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--checkpoint_every_n_examples", type=int, default=1000)


    hparams = parser.parse_args()

    main(hparams)
