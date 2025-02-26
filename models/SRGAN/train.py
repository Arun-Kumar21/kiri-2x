import os
import torch
from torch import nn

from config.config import CONFIG
from data.data_loader import train_loader 

from .discriminator import Discriminator
from .generator import Generator
from .loss import VGGLoss

disc = Discriminator().to(CONFIG.DEVICE)
gen = Generator().to(CONFIG.DEVICE)

criterion = nn.BCELoss()
vgg_loss = VGGLoss()

opt_disc = torch.optim.Adam(disc.parameters(), lr=CONFIG.LEARNING_RATE)
opt_gen = torch.optim.Adam(gen.parameters(), lr=CONFIG.LEARNING_RATE)

for epoch in range(CONFIG.EPOCHS):
    for lr, hr in train_loader:
        lr = lr.to(CONFIG.DEVICE)
        hr = hr.to(CONFIG.DEVICE)

        fake = gen(lr)
        disc_real = disc(hr)
        disc_fake = disc(fake.detach())
        disc_loss_real = criterion(disc_real, torch.ones_like(disc_real) * torch.rand_like(disc_real))

        disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_real + disc_loss_fake

        opt_disc.zero_grad()
        loss_disc.backward()        
        opt_disc.step()

        disc_fake = disc(fake)
        adversial_loss = 1e-3 * criterion(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, hr)
        gen_loss = loss_for_vgg + adversial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()


save_dir = 'weights/'
torch.save(gen.state_dict(), os.join(save_dir, 'SRGAN_gen'))
torch.save(disc.state_dict(), os.join(save_dir, 'SRGAN_disc'))

print("Print Model saved")