from tqdm import tqdm
from src.models.vgg_16 import *
from sewar.full_ref import psnr, mse, vifp
from piq import psnr, ssim, vif, brisque
from frechetdist import frdist
from src.utils.util_data import *


def plot_img(x, pname):
    # x = x.detach().cpu().numpy()
    # x = x[0, 0, :, :]
    plt.imshow(x, cmap='gray')
    plt.title(pname)
    plt.axis('off')
    plt.show()


# GIVEN TWO LISTS COMPUTE THE FRECHET DISTANCE
def f_distance(list1, list2):
    l = [[a, log10(b)] for a, b in zip(range(0, len(list1)), list1)]
    d = [[a, log10(b)] for a, b in zip(range(0, len(list1)), list2)]

    return frdist(l, d)


def calculate_snr(image):
    # Load image and convert to float
    # image = io.imread(image, as_gray=True).astype(float)
    image = conv_to_8bit(image)
    noise_region = image[0:30, 113:143]  # select a patch of 30x30 on the border as noise region

    # Compute mean of signal and noise regions
    signal_region = image[113:143, 113:143]  # select a patch of 30x30 pixels  at the center of the image
    # plot_img(noise_region, '-')
    # plot_img(signal_region, '-')
    signal_mean = np.mean(signal_region)

    noise_std = np.std(noise_region)

    # Compute SNR
    snr = signal_mean / noise_std

    return snr


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# TEST FUNCTION
def test(gen_hd, gen_ld, loader):  # , vgg_ImNet, vgg_tuned, lpips, tag
    gen_hd.eval()
    gen_ld.eval()

    patches_coord = calculate_slice_bboxes(256, 256, 50, 50)
    perc_vgg = lpips(net_type='vgg')
    perc_alex = lpips(net_type='alex')
    perc_squeeze = lpips(net_type='squeeze')
    paq = pyiqa.create_metric('paq2piq', device=torch.device('cpu'), as_loss=False)
    niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
    fid = FrechetInceptionDistance(feature=64)
    kid = KernelInceptionDistance(subset_size=24)
    inception = InceptionScore()

    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        for idx, (LDCT, HDCT, p_id) in enumerate(loop):
            LDCT = LDCT.to(config.DEVICE)
            HDCT = HDCT.to(config.DEVICE)
            fake_HDCT = gen_hd(LDCT)

            # PSNR, SSIM, MSE
            compute_metrics(fake_HDCT[0, 0, :, :].detach().cpu().numpy(), HDCT[0, 0, :, :].detach().cpu().numpy(),
                            p_id[0])

            # VIF
            vif_denoised = vifp(HDCT[0, 0, :, :].detach().cpu().numpy(), fake_HDCT[0, 0, :, :].detach().cpu().numpy())

            # FID, KID, IS
            patches_hd, patches_deno = extract_patches(50, 50, patches_coord,
                                                       HDCT[0, 0, :, :].detach().cpu().numpy(),
                                                       fake_HDCT[0, 0, :, :].detach().cpu().numpy())

            fid.update(patches_hd, real=True)
            fid.update(patches_deno, real=False)
            fid_value = fid.compute().item()

            kid.update(patches_hd, real=True)
            kid.update(patches_deno, real=False)
            kid_value, kid_std = kid.compute()

            inception.update(patches_deno)
            inc_value, inc_std = inception.compute()


            # RAPS-FD
            radial_profile_ld = azimuthalAverage(np.squeeze(LDCT[0, :, :].cpu().detach().numpy()))
            radial_profile_denoised = azimuthalAverage(np.squeeze(fake_HDCT[0, :, :].cpu().detach().numpy()))
            f_d = f_distance(radial_profile_ld, radial_profile_denoised)
            radial_profile_hd = azimuthalAverage(np.squeeze(HDCT[0, :, :].cpu().detach().numpy()))

            # SNR
            snr_ld = calculate_snr(LDCT[0, 0, :, :].detach().cpu().numpy())
            snr_denoised = calculate_snr(fake_HDCT[0, 0, :, :].detach().cpu().numpy())
            snr_hd = calculate_snr(HDCT[0, 0, :, :].detach().cpu().numpy())

            # BRISQUE
            brisque_index = brisque(torch.Tensor(
                np.reshape(normalization(fake_HDCT[0, :, :].cpu().detach().numpy().copy(), range='01'),
                           (1, 1, 256, 256)))).item()


            # LPIPS (VGG, ALEX, SQUEEZE)
            hd_3channel = change_shape_3channels(HDCT.detach().cpu())
            denoised_3channel = change_shape_3channels(fake_HDCT.detach().cpu())
            perceptual_vgg = perc_vgg(hd_3channel, denoised_3channel).item()

            perceptual_alex = perc_alex(hd_3channel, denoised_3channel).item()
            perceptual_squeeze = perc_squeeze(hd_3channel, denoised_3channel).item()

            # PaQ-2-PiQ
            p2p = paq(denoised_3channel).item()

            # NIQE
            niqe_value = niqe(denoised_3channel).item()

            config.raps['ld'][p_id[0]].append(radial_profile_ld.tolist())
            config.raps['hd'][p_id[0]].append(radial_profile_hd.tolist())
            config.raps['denoised'][p_id[0]].append(radial_profile_denoised.tolist())
            config.raps['FD_deno'][p_id[0]].append(f_distance(radial_profile_ld, radial_profile_denoised))
            config.raps['FD_hd'][p_id[0]].append(f_distance(radial_profile_ld, radial_profile_hd))

            config.metrics['SNR_deno'][p_id[0]]['slices'].append(snr_denoised)
            config.metrics['SNR_ld'][p_id[0]]['slices'].append(snr_ld)
            config.metrics['SNR_hd'][p_id[0]]['slices'].append(snr_hd)
            config.metrics['VIF'][p_id[0]]['slices'].append(vif_denoised)
            config.metrics['LPIPS_vgg'][p_id[0]]['slices'].append(perceptual_vgg)
            config.metrics['LPIPS_alex'][p_id[0]]['slices'].append(perceptual_alex)
            config.metrics['LPIPS_squeeze'][p_id[0]]['slices'].append(perceptual_squeeze)
            config.metrics['Brisque'][p_id[0]]['slices'].append(brisque_index)
            config.metrics['PaQ-2-PiQ'][p_id[0]]['slices'].append(p2p)
            config.metrics['FID'][p_id[0]]['slices'].append(fid_value)
            config.metrics['NIQE'][p_id[0]]['slices'].append(niqe_value)
            config.metrics['KID'][p_id[0]]['slices'].append((kid_value.item(), kid_std.item()))
            config.metrics['IS'][p_id[0]]['slices'].append((inc_value.item(), inc_std.item()))

        for i in config.patients:
            config.raps['FD_deno_avg'][i].append(np.mean(config.raps['FD_deno'][i]))
            config.raps['FD_deno_std'][i].append(np.std(config.raps['FD_deno'][i]))
            config.raps['FD_hd_avg'][i].append(np.mean(config.raps['FD_hd'][i]))
            config.raps['FD_hd_std'][i].append(np.std(config.raps['FD_hd'][i]))

        for i in config.metrics.keys():
            for j in config.patients:
                config.metrics[i][j]['avg'].append(np.mean(config.metrics[i][j]['slices']))
                config.metrics[i][j]['std'].append(np.std(config.metrics[i][j]['slices']))



    gen_hd.train()
    gen_ld.train()
