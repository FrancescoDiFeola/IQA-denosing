import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy import fftpack
from src.utils.util_data import *
import pylab as py
from tqdm import tqdm
from frechetdist import frdist

def normalized_power_spectrum(img):
    return np.fft.fft2(img) / np.sqrt(img.shape[0] * img.shape[1])


# COMPUTE RAPS (Radially-averaged-power-spectrum)
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


if __name__ == '__main__':
    dicom_hd = pydicom.read_file(
        '../../data/interim/scapis_10p/10011/tp0a01ed71_ad4c_4025_b8f1_762c47ff73ed_avident2.dcm')
    dicom_ld = pydicom.read_file(
       '../../data/interim/scapis_10p/10011/tp0a01ed71_ad4c_4025_b8f1_762c47ff73ed_avident2.dcm')

    image_hd = transforms_no_bounding(dicom_hd, -500, 1400, tensor_output=False)
    nps_hd = normalized_power_spectrum(image_hd)
    image_ld = transforms_no_bounding(dicom_ld, -500, 1400, tensor_output=False)
    nps_ld = normalized_power_spectrum(image_ld)

    '''print(np.sum(np.abs(image_hd) ** 2))
    print(np.sum(np.abs(nps_hd) ** 2))
    print(np.sum(np.abs(image_ld) ** 2))
    print(np.sum(np.abs(nps_ld) ** 2))

    plt.imshow(nps_hd)
    plt.show()
    plt.imshow(image_ld)
    plt.show()'''

    F_ld = fftpack.fftshift(fftpack.fft2(image_ld))
    F_hd = fftpack.fftshift(fftpack.fft2(image_hd))

    psd2D_ld = np.abs(F_ld) ** 2
    psd2D_hd = np.abs(F_hd) ** 2
    psd1d_ld = azimuthalAverage(psd2D_ld)
    psd1d_hd = azimuthalAverage(psd2D_hd)
    # profiles_27 = open(f'../../reports/radial_profiles/radial_profiles_exp27_ep200_per_patient.json')
    # profiles_27 = json.load(profiles_27)
    # profiles_24 = open(f'../../reports/radial_profiles/radial_profiles_exp24_ep200_per_patient.json')
    # profiles_24 = json.load(profiles_24)
    # profiles_28 = open(f'../../reports/radial_profiles/radial_profiles_exp28_ep200_per_patient.json')
    # profiles_28 = json.load(profiles_28)
    # profiles_29 = open(f'../../reports/radial_profiles/radial_profiles_exp29_ep200_per_patient.json')
    # profiles_29 = json.load(profiles_29)
    # profiles_30 = open(f'../../reports/radial_profiles/radial_profiles_exp30_ep200_per_patient.json')
    # profiles_30 = json.load(profiles_30)
    # profiles_31 = open(f'../../reports/radial_profiles/radial_profiles_exp31_ep200_per_patient.json')
    # profiles_31 = json.load(profiles_31)
    # profiles_32 = open(f'../../reports/radial_profiles/radial_profiles_exp32_ep200_per_patient.json')
    # profiles_32 = json.load(profiles_32)
    # profiles_33 = open(f'../../reports/radial_profiles/radial_profiles_exp33_ep200_per_patient.json')
    # profiles_33 = json.load(profiles_33)

    '''mean_profile_ld = [np.mean(k) for k in zip(*profiles_27['ld'])]
    mean_profile_denoised_27 = [np.mean(k) for k in zip(*profiles_27['denoised'])]
    mean_profile_denoised_28 = [np.mean(k) for k in zip(*profiles_28['denoised'])]
    mean_profile_denoised_29 = [np.mean(k) for k in zip(*profiles_29['denoised'])]
    mean_profile_denoised_24 = [np.mean(k) for k in zip(*profiles_24['denoised'])]
    mean_profile_denoised_30 = [np.mean(k) for k in zip(*profiles_30['denoised'])]
    mean_profile_denoised_31 = [np.mean(k) for k in zip(*profiles_31['denoised'])]
    mean_profile_denoised_32 = [np.mean(k) for k in zip(*profiles_32['denoised'])]
    mean_profile_denoised_33 = [np.mean(k) for k in zip(*profiles_33['denoised'])]

    psd1d_ld = mean_profile_ld
    psd1d_hd_27 = mean_profile_denoised_27
    psd1d_hd_24 = mean_profile_denoised_24
    psd1d_hd_28 = mean_profile_denoised_28
    psd1d_hd_29 = mean_profile_denoised_29
    psd1d_hd_30 = mean_profile_denoised_30
    psd1d_hd_31 = mean_profile_denoised_31
    psd1d_hd_32 = mean_profile_denoised_32
    psd1d_hd_33 = mean_profile_denoised_33

    py.figure()
    py.clf()
    py.semilogy(psd1d_ld)
    py.semilogy(psd1d_hd_32)
    py.semilogy(psd1d_hd_30)
    py.semilogy(psd1d_hd_27)
    py.semilogy(psd1d_hd_24)
    py.semilogy(psd1d_hd_28)
    py.semilogy(psd1d_hd_29)
    py.semilogy(psd1d_hd_33)
    py.semilogy(psd1d_hd_31)

    py.xlabel('Spatial Frequency')
    py.ylabel('Power Spectrum')
    py.legend(['ld', 'cyc', 'cyc+id', 'cyc+har', 'cyc+id+har', 'cyc+perc(tuned)', 'cyc+id+perc(tuned)', 'cyc+perc(ImNet)', 'cyc+id+perc(ImNet)'])
    py.savefig(f'../../reports/plots/radial_profiles_lung_pilot_tot_8exp.png')
    py.show()'''


    # ---------------------------------------------------------------------------------------------------------------------------
    # compute the average profile per patient
    def avg_std(value, l):
        print(len(l))
        return np.sqrt(np.sum(np.square(value)) / len(l))


    def compute_avg_profiles(profile, exp):

        denoised_avg = {'denoised_avg': []}
        denoised_std = {'denoised_std': []}
        patients_id = [10011, 10019, 10028, 10033, 10038, 10039, 10040, 10041, 10046, 10055, 10062, 10068,
                       10075, 10096, 10247, 10267, 10302, 10345, 10355, 10374, 10380, 10390, 10398, 10413,
                       10453, 10454, 10495, 10505, 10506, 10514, 10515, 10522, 10523, 10537, 10546, 10547,
                       10551, 10573, 10578, 10583, 10590, 10595, 10597, 10614, 10624, 10651, 10657, 10660,
                       10674, 10704, 10709, 10712, 10714, 10725, 10735, 11358, 11395, 11402, 11418, 11447,
                       11454, 11458, 11473, 11494, 11498, 11505, 11507, 11509, 11516, 11518, 11538, 11562,
                       11563, 11577, 11578, 11599, 11608, 11610, 11624, 11629, 11634, 11638, 11652, 12860,
                       12865, 12887, 12891, 12900, 12901, 12912, 12918, 12987, 13014, 13026, 13077, 13078,
                       13090, 13099, 13111, 13129, 13131, 13133, 13137, 13145, 13147, 13152, 13156, 13157,
                       13159, 13175, 13190, 13218, 13225, 13272, 13329, 13340, 13347, 13362, 13363, 13369,
                       13375, 13380, 13401, 13413, 13425, 13431, 13433, 13457, 13464, 13467, 13485, 13490,
                       13525, 13534, 13536, 13546, 13551, 13565, 13571, 13583, 13585, 13598, 13604, 13611,
                       13619, 13643, 13658, 13661, 13690, 13696, 13725, 13729, 13751, 14063, 14065, 14071,
                       14072, 14087, 14111, 14148, 14156, 14158, 14173, 14188, 14200, 14203, 14219, 14222,
                       14226, 14227, 14229, 14232, 14233, 14249, 14264, 14282, 14294, 14324, 14334, 14337,
                       14339, 14349, 14352, 14371, 14405, 14682, 14723, 14732, 14739, 14745, 14747, 14801,
                       14812, 14832]

        for i in tqdm(patients_id):
            average_per_patient_profile_denoised = [np.mean(k) for k in zip(*profile['denoised'][str(i)])]
            std_per_patient_profile_denoised = [np.std(k) for k in zip(*profile['denoised'][str(i)])]
            denoised_avg['denoised_avg'].append(average_per_patient_profile_denoised)
            denoised_std['denoised_std'].append(std_per_patient_profile_denoised)

        denoised_avg_total = [np.mean(k) for k in zip(*denoised_avg['denoised_avg'])]
        denoised_std_total = [avg_std(k, denoised_avg['denoised_avg']) for k in zip(*denoised_std['denoised_std'])]

        denoised_avg = json.dumps(denoised_avg, indent=6)
        with open(f"../../reports/radial_profiles/denoised_avg_per_pat_{exp}.json", 'w') as f:
            f.write(denoised_avg)

        denoised_std = json.dumps(denoised_std, indent=6)
        with open(f"../../reports/radial_profiles/denoised_std_per_pat_{exp}.json", 'w') as f:
            f.write(denoised_std)

        denoised_avg_total = json.dumps(denoised_avg_total, indent=6)
        with open(f"../../reports/radial_profiles/denoised_avg_tot_{exp}.json", 'w') as f:
            f.write(denoised_avg_total)
        denoised_std_total = json.dumps(denoised_std_total, indent=6)
        with open(f"../../reports/radial_profiles/denoised_std_tot_{exp}.json", 'w') as f:
            f.write(denoised_std_total)


    '''compute_avg_profiles(profiles_27, 27)
    compute_avg_profiles(profiles_28, 28)
    compute_avg_profiles(profiles_29, 29)
    compute_avg_profiles(profiles_30, 30)
    compute_avg_profiles(profiles_31, 31)
    compute_avg_profiles(profiles_32, 32)
    compute_avg_profiles(profiles_33, 33)'''
    '''ld_avg_24 = {'ld_avg': []}
    ld_std_24 = {'ld_std': []}
    denoised_avg_24 = {'denoised_avg': []}
    denoised_std_24 = {'denoised_std': []}
    patients_id = [10011, 10019, 10028, 10033, 10038, 10039, 10040, 10041, 10046, 10055, 10062, 10068,
                   10075, 10096, 10247, 10267, 10302, 10345, 10355, 10374, 10380, 10390, 10398, 10413,
                   10453, 10454, 10495, 10505, 10506, 10514, 10515, 10522, 10523, 10537, 10546, 10547,
                   10551, 10573, 10578, 10583, 10590, 10595, 10597, 10614, 10624, 10651, 10657, 10660,
                   10674, 10704, 10709, 10712, 10714, 10725, 10735, 11358, 11395, 11402, 11418, 11447,
                   11454, 11458, 11473, 11494, 11498, 11505, 11507, 11509, 11516, 11518, 11538, 11562,
                   11563, 11577, 11578, 11599, 11608, 11610, 11624, 11629, 11634, 11638, 11652, 12860,
                   12865, 12887, 12891, 12900, 12901, 12912, 12918, 12987, 13014, 13026, 13077, 13078,
                   13090, 13099, 13111, 13129, 13131, 13133, 13137, 13145, 13147, 13152, 13156, 13157,
                   13159, 13175, 13190, 13218, 13225, 13272, 13329, 13340, 13347, 13362, 13363, 13369,
                   13375, 13380, 13401, 13413, 13425, 13431, 13433, 13457, 13464, 13467, 13485, 13490,
                   13525, 13534, 13536, 13546, 13551, 13565, 13571, 13583, 13585, 13598, 13604, 13611,
                   13619, 13643, 13658, 13661, 13690, 13696, 13725, 13729, 13751, 14063, 14065, 14071,
                   14072, 14087, 14111, 14148, 14156, 14158, 14173, 14188, 14200, 14203, 14219, 14222,
                   14226, 14227, 14229, 14232, 14233, 14249, 14264, 14282, 14294, 14324, 14334, 14337,
                   14339, 14349, 14352, 14371, 14405, 14682, 14723, 14732, 14739, 14745, 14747, 14801,
                   14812, 14832]

    for i in tqdm(patients_id):
        average_per_patient_profile_ld_24 = [np.mean(k) for k in zip(*profiles_24['ld'][str(i)])]
        std_per_patient_profile_ld_24 = [np.std(k) for k in zip(*profiles_24['ld'][str(i)])]
        ld_avg_24['ld_avg'].append(average_per_patient_profile_ld_24)
        ld_std_24['ld_std'].append(std_per_patient_profile_ld_24)

        average_per_patient_profile_denoised_24 = [np.mean(k) for k in zip(*profiles_24['denoised'][str(i)])]
        std_per_patient_profile_denoised_24 = [np.std(k) for k in zip(*profiles_24['denoised'][str(i)])]
        denoised_avg_24['denoised_avg'].append(average_per_patient_profile_denoised_24)
        denoised_std_24['denoised_std'].append(std_per_patient_profile_denoised_24)

    def avg_std(value, l):
        print(len(l))
        return np.sqrt(np.sum(np.square(value))/len(l))


    ld_avg_total = [np.mean(k) for k in zip(*ld_avg_24['ld_avg'])]
    ld_std_total = [avg_std(k, ld_avg_24['ld_avg']) for k in zip(*ld_std_24['ld_std'])]

    denoised_avg_total_24 = [np.mean(k) for k in zip(*denoised_avg_24['denoised_avg'])]
    denoised_std_total_24 = [avg_std(k, ld_avg_24['ld_avg']) for k in zip(*denoised_std_24['denoised_std'])]


    ld_avg = json.dumps(ld_avg_24, indent=6)
    with open(f"../../reports/radial_profiles/ld_avg_per_pat.json", 'w') as f:
        f.write(ld_avg)

    ld_std = json.dumps(ld_std_24, indent=6)
    with open(f"../../reports/radial_profiles/ld_std_per_pat.json", 'w') as f:
        f.write(ld_std)

    denoised_avg = json.dumps(denoised_avg_24, indent=6)
    with open(f"../../reports/radial_profiles/denoised_avg_per_pat_24.json", 'w') as f:
        f.write(denoised_avg)

    denoised_std = json.dumps(denoised_std_24, indent=6)
    with open(f"../../reports/radial_profiles/denoised_std_per_pat_24.json", 'w') as f:
        f.write(denoised_std)

    ld_avg_tot = json.dumps(ld_avg_total, indent=6)
    with open(f"../../reports/radial_profiles/ld_avg_tot.json", 'w') as f:
        f.write(ld_avg_tot)
    ld_std_tot = json.dumps( ld_std_total, indent=6)
    with open(f"../../reports/radial_profiles/ld_std_tot.json", 'w') as f:
            f.write(ld_std_tot)

    denoised_avg_tot = json.dumps(denoised_avg_total_24, indent=6)
    with open(f"../../reports/radial_profiles/denoised_avg_tot_24.json", 'w') as f:
        f.write(denoised_avg_tot)
    denoised_std_tot = json.dumps(denoised_std_total_24, indent=6)
    with open(f"../../reports/radial_profiles/denoised_std_tot_24.json", 'w') as f:
            f.write(denoised_std_tot)'''
    # ---------------------------------------------------------------------
    ld_avg = open(f'../../reports/radial_profiles/ld_avg_tot.json')
    ld_avg = json.load(ld_avg)
    ld_std = open(f'../../reports/radial_profiles/ld_std_tot.json')
    ld_std = json.load(ld_std)
    denoised_avg_24 = open(f'../../reports/radial_profiles/denoised_avg_tot_24.json')
    denoised_avg_24 = json.load(denoised_avg_24)

    denoised_std_24 = open(f'../../reports/radial_profiles/denoised_std_tot_24.json')
    denoised_std_24 = json.load(denoised_std_24)
    denoised_avg_27 = open(f'../../reports/radial_profiles/denoised_avg_tot_27.json')
    denoised_avg_27 = json.load(denoised_avg_27)
    denoised_std_27 = open(f'../../reports/radial_profiles/denoised_std_tot_27.json')
    denoised_std_27 = json.load(denoised_std_27)
    denoised_avg_28 = open(f'../../reports/radial_profiles/denoised_avg_tot_28.json')
    denoised_avg_28 = json.load(denoised_avg_28)
    denoised_std_28 = open(f'../../reports/radial_profiles/denoised_std_tot_28.json')
    denoised_std_28 = json.load(denoised_std_28)
    denoised_avg_29 = open(f'../../reports/radial_profiles/denoised_avg_tot_29.json')
    denoised_avg_29 = json.load(denoised_avg_29)
    denoised_std_29 = open(f'../../reports/radial_profiles/denoised_std_tot_29.json')
    denoised_std_29 = json.load(denoised_std_29)
    denoised_avg_30 = open(f'../../reports/radial_profiles/denoised_avg_tot_30.json')
    denoised_avg_30 = json.load(denoised_avg_30)
    denoised_std_30 = open(f'../../reports/radial_profiles/denoised_std_tot_30.json')
    denoised_std_30 = json.load(denoised_std_30)
    denoised_avg_31 = open(f'../../reports/radial_profiles/denoised_avg_tot_31.json')
    denoised_avg_31 = json.load(denoised_avg_31)
    denoised_std_31 = open(f'../../reports/radial_profiles/denoised_std_tot_31.json')
    denoised_std_31 = json.load(denoised_std_31)
    denoised_avg_32 = open(f'../../reports/radial_profiles/denoised_avg_tot_32.json')
    denoised_avg_32 = json.load(denoised_avg_32)
    denoised_std_32 = open(f'../../reports/radial_profiles/denoised_std_tot_32.json')
    denoised_std_32 = json.load(denoised_std_32)
    denoised_avg_33 = open(f'../../reports/radial_profiles/denoised_avg_tot_33.json')
    denoised_avg_33 = json.load(denoised_avg_33)
    denoised_std_33 = open(f'../../reports/radial_profiles/denoised_std_tot_33.json')
    denoised_std_33 = json.load(denoised_std_33)

    negative_std_ld = [a_i - b_i for a_i, b_i in zip(ld_avg, ld_std)]
    positive_std_ld = [a_i + b_i for a_i, b_i in zip(ld_avg, ld_std)]

    negative_std_denoised_24 = [a_i - b_i for a_i, b_i in zip(denoised_avg_24, denoised_std_24)]
    positive_std_denoised_24 = [a_i + b_i for a_i, b_i in zip(denoised_avg_24, denoised_std_24)]
    negative_std_denoised_27 = [a_i - b_i for a_i, b_i in zip(denoised_avg_27, denoised_std_27)]
    positive_std_denoised_27 = [a_i + b_i for a_i, b_i in zip(denoised_avg_27, denoised_std_27)]
    negative_std_denoised_28 = [a_i - b_i for a_i, b_i in zip(denoised_avg_28, denoised_std_28)]
    positive_std_denoised_28 = [a_i + b_i for a_i, b_i in zip(denoised_avg_28, denoised_std_28)]
    negative_std_denoised_29 = [a_i - b_i for a_i, b_i in zip(denoised_avg_29, denoised_std_29)]
    positive_std_denoised_29 = [a_i + b_i for a_i, b_i in zip(denoised_avg_29, denoised_std_29)]
    negative_std_denoised_30 = [a_i - b_i for a_i, b_i in zip(denoised_avg_30, denoised_std_30)]
    positive_std_denoised_30 = [a_i + b_i for a_i, b_i in zip(denoised_avg_30, denoised_std_30)]
    negative_std_denoised_31 = [a_i - b_i for a_i, b_i in zip(denoised_avg_31, denoised_std_31)]
    positive_std_denoised_31 = [a_i + b_i for a_i, b_i in zip(denoised_avg_31, denoised_std_31)]
    negative_std_denoised_32 = [a_i - b_i for a_i, b_i in zip(denoised_avg_32, denoised_std_32)]
    positive_std_denoised_32 = [a_i + b_i for a_i, b_i in zip(denoised_avg_32, denoised_std_32)]
    negative_std_denoised_33 = [a_i - b_i for a_i, b_i in zip(denoised_avg_33, denoised_std_33)]
    positive_std_denoised_33 = [a_i + b_i for a_i, b_i in zip(denoised_avg_33, denoised_std_33)]

    py.figure()
    py.clf()
    '''py.subplot(4, 1, 1)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_32, positive_std_denoised_32, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_32, color='green')
    py.grid()
    py.legend(['ld', 'L_adv+L_cyc'])
    py.ylabel('Power Spectrum')
    py.subplot(4, 1, 2)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_30, positive_std_denoised_30, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_30, color='green')
    py.grid()
    py.legend(['ld', 'L_adv+L_cyc+L_id'])
    py.ylabel('Power Spectrum')
    py.subplot(4, 1, 3)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_27, positive_std_denoised_27, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_27, color='green')
    py.grid()
    py.legend(['ld', 'L_adv+L_cyc+L_h'])
    py.ylabel('Power Spectrum')
    py.subplot(4, 1, 4)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_24, positive_std_denoised_24, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_24, color='green')
    py.grid()
    py.legend(['ld', 'L_adv+L_cyc+L_id+L_h'])
    py.ylabel('Power Spectrum')'''

    py.subplot(4, 1, 1)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_28, positive_std_denoised_28, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_28, color='green')
    py.grid()
    py.legend(['ld', 'L_adv+L_cyc+Lp_tuned'])
    py.ylabel('Power Spectrum')
    py.subplot(4, 1, 2)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_29, positive_std_denoised_29, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_29, color='green')
    py.grid()
    py.legend(['ld', 'L_adv+L_cyc+L_id+Lp_tuned'])
    py.ylabel('Power Spectrum')
    py.subplot(4, 1, 3)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_33, positive_std_denoised_33, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_33, color='green')
    py.xlabel('Spatial Frequency')
    py.legend(['ld', 'L_adv+Lcyc+Lp_ImNet'])
    py.grid()
    py.ylabel('Power Spectrum')
    py.subplot(4, 1, 4)
    py.fill_between(range(0, len(ld_avg)), negative_std_ld, positive_std_ld, linestyle='None', label='_nolegend_')
    py.semilogy(ld_avg, color='red')
    py.fill_between(range(0, len(ld_avg)), negative_std_denoised_31, positive_std_denoised_31, linestyle='None', label='_nolegend_')
    py.semilogy(denoised_avg_31, color='green')
    py.xlabel('Spatial Frequency')
    py.legend(['ld', 'L_adv+L_cyc+L_id+Lp_ImNet'])
    py.grid()
    py.ylabel('Power Spectrum')
    py.show()

    py.figure()
    py.clf()
    py.semilogy(ld_avg)
    py.semilogy(denoised_avg_32)
    py.semilogy(denoised_avg_30)
    py.semilogy(denoised_avg_27)
    py.semilogy(denoised_avg_24)
    py.semilogy(denoised_avg_28)
    py.semilogy(denoised_avg_29)
    py.semilogy(denoised_avg_33)
    py.semilogy(denoised_avg_31)

    py.xlabel('Spatial Frequency')
    py.ylabel('Power Spectrum')
    py.grid()
    py.legend(
        ['low-dose', 'L_adv+L_cyc', 'L_adv+L_cyc+L_id', 'L_adv+L_cyc+L_h', 'L_adv+L_cyc+L_id+L_h', 'L_adv+L_cyc+Lp_tuned', 'L_adv+L_cyc+L_id+Lp_tuned', 'L_adv+Lcyc+Lp_ImNet',
         'L_adv+L_cyc+L_id+Lp_ImNet'])
    py.show()

    # percentage difference
    # ld_avg: low_dose spectrum, denoised_avg_[]: i-esimo experiment spectrum
    avg_decrease_32 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_32))/np.array(ld_avg))
    avg_decrease_30 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_30)) / np.array(ld_avg))
    avg_decrease_27 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_27)) / np.array(ld_avg))
    avg_decrease_24 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_24)) / np.array(ld_avg))
    avg_decrease_28 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_28)) / np.array(ld_avg))
    avg_decrease_29 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_29)) / np.array(ld_avg))
    avg_decrease_33 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_33)) / np.array(ld_avg))
    avg_decrease_31 = np.mean(abs(np.array(ld_avg) - np.array(denoised_avg_31)) / np.array(ld_avg))
    print(avg_decrease_32,avg_decrease_30, avg_decrease_27, avg_decrease_24, avg_decrease_28, avg_decrease_29, avg_decrease_33, avg_decrease_31)
