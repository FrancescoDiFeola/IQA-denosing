import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips
from src.models.discriminator_model import Discriminator
from src.models.generator_model import Generator
from src.models.vgg_16 import *
from src.data.plot_functions import *
from src.utils.utils_checkpoints import save_checkpoint, load_checkpoint, seed_everything
from test_function import test
from train import train
from validation import validation
import time
from src.losses.coweighting_denoGAN import CoVWeightingLoss
import pandas as pd

def main():
    disc_HD = Discriminator(in_channels=1).to(config.DEVICE)
    disc_LD = Discriminator(in_channels=1).to(config.DEVICE)
    gen_LD = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)  # generator for LDCT
    gen_HD = Generator(img_channels=1, num_residuals=9).to(config.DEVICE)  # generators for HDCT

   
    # OPTIMIZER DISCRIMINATOR
    opt_disc = optim.Adam(
        list(disc_HD.parameters()) + list(disc_LD.parameters()),  # concatenazione dei parametri dei due discriminatori
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),  # values from the paper
    )

    # OPTIMIZER GENERATOR
    opt_gen = optim.Adam(
        list(gen_LD.parameters()) + list(gen_HD.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()  # L1 loss for the cycle consistency loss and identity
    mse = nn.MSELoss()  # adversarial loss

    # load the checkpoints to run a previously trained model
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_HD, gen_HD, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_LD, gen_LD, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_HD, disc_HD, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_LD, disc_LD, opt_disc, config.LEARNING_RATE,
        )

    # DATASETS
    training_dataset = Dataset(annotations=config.annotation_training_mayo_baseline, centre=-500, width=1400, segmentation=False)

    train_loader = DataLoader(
        training_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_dataset = Dataset(annotations=config.val_and_test, centre=-500, width=1400, segmentation=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )



    # training loop
    for epoch in range(config.NUM_EPOCHS):
        '''tic_1 = time.time()
        train(disc_HD, disc_LD, gen_LD, gen_HD, validation_loader, opt_disc, opt_gen, L1, mse, vgg_tuned, coweighting)
        toc_1 = time.time()'''

        # CHECKPOINTS
        if config.SAVE_MODEL and ((epoch % 20 == 0) or epoch == 199):
            save_checkpoint(gen_HD, opt_gen,
                            filename=f'../../models/training_{config.CURRENT_EXP}/gen_hd_{epoch}.pth.tar')
            save_checkpoint(gen_LD, opt_gen,
                            filename=f'../../models/training_{config.CURRENT_EXP}/gen_ld_{epoch}.pth.tar')
            save_checkpoint(disc_HD, opt_disc,
                            filename=f'../../models/training_{config.CURRENT_EXP}/disc_hd_{epoch}.pth.tar')
            save_checkpoint(disc_LD, opt_disc,
                            filename=f'../../models/training_{config.CURRENT_EXP}/disc_ld_{epoch}.pth.tar')

        # ---------------------------------------------------------------------------------------------------------------


        test(gen_HD, gen_LD, test_loader)  # , vgg_ImNet, vgg_tuned, lpips




        metrics = json.dumps(config.metrics, indent=6)
        with open(f"../../reports/metrics/metrics.json", 'w') as f:
             f.write(metrics)

        raps = json.dumps(config.raps, indent=6)
        with open(f"../../reports/metrics/raps.json", 'w') as f:
             f.write(raps)



if __name__ == "__main__":
    seed_everything()
    main()
