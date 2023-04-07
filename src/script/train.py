from src.utils.utils_checkpoints import save_checkpoint, load_checkpoint, seed_everything
from torch.utils.data import DataLoader
import torch.optim as optim
import src.utils.util_general as config
from tqdm import tqdm
from src.models.vgg_16 import *
from src.models.discriminator_model import Discriminator
from src.models.generator_model import Generator
from src.models.vgg_16 import *
import json
from torchvision.utils import save_image
from src.losses.coweighting_denoGAN import CoVWeightingLoss
from src.data.plot_functions import *


def train(disc_hd, disc_ld, gen_ld, gen_hd, loader, opt_disc, opt_gen, l1, mse, vgg, covweighting):
    loop = tqdm(loader, leave=True)

    for idx, (LDCT, HDCT) in enumerate(loop):
        LDCT = LDCT.to(config.DEVICE)  # zebra=zebra.to('gpu:0')  noisy image
        HDCT = HDCT.to(config.DEVICE)

        # Training for the discriminators
        # with torch.cuda.amp.autocast():  # ?? (necessary for float16)
        fake_HDCT = gen_hd(LDCT)  # denoised image
        D_HD_real = disc_hd(HDCT)
        D_HD_fake = disc_hd(fake_HDCT)  # disc_H(fake_horse.detach()), we put detach here because we are going to
        # use the 'fake_horse' when we train the generator and doing detach we don't have to repeat
        # fake_horse = gen_H(zebra) when we train the generator.

        D_HD_real_loss = mse(D_HD_real, torch.ones_like(D_HD_real))
        D_HD_fake_loss = mse(D_HD_fake, torch.zeros_like(D_HD_fake))
        D_HD_loss = D_HD_real_loss + D_HD_fake_loss

        fake_LDCT = gen_ld(HDCT)
        D_LD_real = disc_ld(LDCT)
        D_LD_fake = disc_ld(fake_LDCT)  # disc_Z(fake_zebra.detach())

        D_LD_real_loss = mse(D_LD_real, torch.ones_like(D_LD_real))
        D_LD_fake_loss = mse(D_LD_fake, torch.zeros_like(D_LD_fake))
        D_LD_loss = D_LD_real_loss + D_LD_fake_loss

        D_loss = (D_HD_loss + D_LD_loss) / 2  # discriminator loss

        opt_disc.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_disc.step()

        # Training for the generators
        # with torch.cuda.amp.autocast():

        # adversarial loss for both generators
        D_HD_fake = disc_hd(fake_HDCT)
        D_LD_fake = disc_ld(fake_LDCT)
        loss_G_HD = mse(D_HD_fake, torch.ones_like(D_HD_fake))
        loss_G_LD = mse(D_LD_fake, torch.ones_like(D_LD_fake))

        # cycle loss
        cycle_LDCT = gen_ld(fake_HDCT)
        cycle_HDCT = gen_hd(fake_LDCT)
        cycle_LDCT_loss = l1(LDCT, cycle_LDCT)
        cycle_HDCT_loss = l1(HDCT, cycle_HDCT)

        # identity loss
        identity_LDCT = gen_ld(LDCT)
        identity_HDCT = gen_hd(HDCT)
        identity_LDCT_loss = l1(LDCT, identity_LDCT)
        identity_HDCT_loss = l1(HDCT, identity_HDCT)

        # perceptual loss
        perceptual_loss = perceptual_similarity_loss(gen_hd, gen_ld, LDCT, HDCT, vgg)

        # Haralick loss
        loss_h1_HDCT = compute_haralick(cycle_HDCT[0, 0, :, :].detach().cpu().numpy(),
                                        HDCT[0, 0, :, :].detach().cpu().numpy())
        loss_h1_LDCT = compute_haralick(cycle_LDCT[0, 0, :, :].detach().cpu().numpy(),
                                        LDCT[0, 0, :, :].detach().cpu().numpy())

        print(f'haralick: {loss_h1_HDCT + loss_h1_LDCT}')
        print(f'perceptual_loss: {perceptual_loss}')
        print(f'identity loss: {identity_LDCT_loss + identity_HDCT_loss}')
        print(f'cycle loss: {cycle_LDCT_loss + cycle_HDCT_loss}')
        print(f'adversarial loss: {loss_G_LD + loss_G_HD}')

        # add all together
        '''G_loss = (
                loss_G_LD
                + loss_G_HD
                + cycle_LDCT_loss * config.LAMBDA_CYCLE
                + cycle_HDCT_loss * config.LAMBDA_CYCLE
                + identity_HDCT_loss * config.LAMBDA_IDENTITY
                + identity_LDCT_loss * config.LAMBDA_IDENTITY
                # + perceptual_loss
                + loss_h1_HDCT
                + loss_h1_LDCT
        )'''

        G_loss_list = [(loss_G_LD + loss_G_HD), (cycle_LDCT_loss + cycle_HDCT_loss),
                       (identity_HDCT_loss + identity_LDCT_loss), perceptual_loss, (loss_h1_HDCT + loss_h1_LDCT)]

        G_loss_weighted = covweighting(G_loss_list)
        opt_gen.zero_grad()
        G_loss_weighted.backward()
        opt_gen.step()

    config.training_losses['D_total_loss'].append(D_loss.item())
    config.training_losses['cycle_loss'].append((cycle_LDCT_loss + cycle_HDCT_loss).item())
    config.training_losses['Haralick_loss'].append((loss_h1_HDCT + loss_h1_LDCT).item())
    config.training_losses['Perceptual_loss'].append(perceptual_loss.item())
    config.training_losses['identity_loss'].append((identity_HDCT_loss + identity_LDCT_loss).item())
