import numpy as np
import pandas as pd
import torch
import os
import json

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 10
LAMBDA_ADVERSARIAL = 1
LAMBDA_IDENTITY = 0  # 0.5 * LAMBDA_CYCLE
LAMBDA_PERCEPTUAL = 1
NUM_WORKERS = 4
NUM_EPOCHS = 1
NUM_LOSSES = 0

# CHECKPOINTS
LOAD_MODEL = True
SAVE_MODEL = False

CURRENT_EXP = 30
CURRENT_EPOCH = 200
CHECKPOINT_GEN_HD = f"../../models/models_tr_{CURRENT_EXP}/gen_hd_{CURRENT_EPOCH}.pth.tar"
CHECKPOINT_GEN_LD = f"../../models/models_tr_{CURRENT_EXP}/gen_ld_{CURRENT_EPOCH}.pth.tar"
CHECKPOINT_DISC_HD = f"../../models/models_tr_{CURRENT_EXP}/disc_hd_{CURRENT_EPOCH}.pth.tar"
CHECKPOINT_DISC_LD = f"../../models/models_tr_{CURRENT_EXP}/disc_ld_{CURRENT_EPOCH}.pth.tar"

# ANNOTATION FILES
annotations = '../data/csv_files/LIDC_reduced.csv'
mayo_training = '../data/csv_files/training_mayo_shuffled.csv'
mayo_training_extended = '../data/csv_files/mayo_extended_updated_complete.csv'
annotations_test = '../data/csv_files/mayo_test_annotation_file.csv'
annotations_test_mayo_p1 = '../data/csv_files/test_mayo_pt1.csv'
annotations_test_mayo_p1_segmented = '../data/cvs_files/test_mayo_pt1_segmentation.csv'
annotation_training_mayo_baseline = '../data/csv_files/training_mayo_baseline.csv'
annotation_validation = '../data/validation.csv'
validation = '../data/validation_5_patients.csv'
test = '../data/test_5_patients_sg.csv'
scapis_10p = '../data/scapis_10p.csv'
lung_pilot = '../data/lung_pilot.csv'
scapis_20p = '../data/scapis_20p_segm.csv'
val_and_test = '../data/10_patients_test_segm.csv'

# STORAGE FOR PARAMETERS
training_losses = {'D_total_loss': [],
                   'cycle_loss': [],
                   'identity_loss': [],
                   'Perceptual_loss': [],
                   'Haralick_loss': [],
                   }

losses_batch = {'D_total_loss': [],
                'cycle_loss': [],
                'identity_loss': [],
                'Perceptual_loss': [],
                'Haralick_loss': [],
                }

validation_losses = {'D_total_loss': [],
                     'cycle_loss': [],
                     'Perceptual_loss': [],
                     'identity_loss': [],
                     'Haralick_loss': [],
                     }

radial_profiles = {'ld': [],
                   'denoised': [],
                   }

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

radial_profiles_per_patient = {'ld': {j: [] for j in patients_id},
                               'denoised': {j: [] for j in patients_id},
                               }

psnr = []
mse = []
ssim = []
ifc = []
lpips = []
vgg_pre = []
vgg_tuned = []
snr_ld = []
snr_denoised = []
mutual_info = []
brisque_ld = []
brisque_denoised = []


patients = ['C030', 'C120', 'C124', 'C162', 'C170', 'C052', 'C067', 'C130', 'C166', 'C135']
raps = {'ld': {j: [] for j in patients},
        'denoised': {j: [] for j in patients},
        'hd': {j: [] for j in patients},
        'FD_deno': {j: [] for j in patients},
        'FD_deno_avg': {j: [] for j in patients},
        'FD_deno_std': {j: [] for j in patients},
        'FD_hd': {j: [] for j in patients},
        'FD_hd_avg': {j: [] for j in patients},
        'FD_hd_std': {j: [] for j in patients},
        }

values = ['slices', 'avg', 'std']
metrics = {'PSNR': {pat: {j: [] for j in values} for pat in patients},
           'MSE': {pat: {j: [] for j in values} for pat in patients},
           'SSIM':  {pat: {j: [] for j in values} for pat in patients},
           'VIF': {pat: {j: [] for j in values} for pat in patients},
           'SNR_deno': {pat: {j: [] for j in values} for pat in patients},
           'SNR_ld': {pat: {j: [] for j in values} for pat in patients},
           'SNR_hd': {pat: {j: [] for j in values} for pat in patients},
           'FID': {pat: {j: [] for j in values} for pat in patients},
           'IS': {pat: {j: [] for j in values} for pat in patients},
           'KID': {pat: {j: [] for j in values} for pat in patients},
           'LPIPS_vgg': {pat: {j: [] for j in values} for pat in patients},
           'LPIPS_alex': {pat: {j: [] for j in values} for pat in patients},
           'LPIPS_squeeze': {pat: {j: [] for j in values} for pat in patients},
           'Brisque': {pat: {j: [] for j in values} for pat in patients},
           'PaQ-2-PiQ': {pat: {j: [] for j in values} for pat in patients},
           'NIQE':  {pat: {j: [] for j in values} for pat in patients},
           }


regression_scores = {
    'MSE_avg': [],
    'MSE_std': [],
    'feature_importance': [],
}

values = ['avg', 'std']
metrics_validation = {'PSNR': {j: [] for j in values},
                      'SSIM': {j: [] for j in values},
                      'IFC': {j: [] for j in values},
                      'LPIPS': {j: [] for j in values},
                      'vgg_pre': {j: [] for j in values},
                      'vgg_tuned': {j: [] for j in values},
                      }

modality = ['LD', 'denoised']
unpaired_metrics = {
    'SNR': {j: [] for j in modality},
    'M.I.': {j: [] for j in modality},
    'BRISQUE': {j: [] for j in modality},

}

color_legend_unpaired_features = {
                         'FID': 'grey',
                         'KID': 'pink',
                         'IS': 'black',
                         'BRISQUE': 'green',
                         'SNR': 'blue',
                         'FD-deno': 'yellow',
                         'PaQ-2-PiQ': 'purple',
                         'NIQE': 'brown',
                         }


anular_circles_paired_labels = {
    '1': [],
    '2': [],
    '3': [],
    '4': [],
    '5': [],
    '6': [],
    '7': [],
    '8': [],
}

color_legend_paired_features = {'PSNR': 'green',
                       'MSE': 'blue',
                       'SSIM': 'yellow',
                       'VIF': 'purple',
                       'LPIPS-1': 'brown',
                       'LPIPS-2': 'red',
                       'LPIPS-3': 'orange',
                       }

anular_circles_unpaired_labels = {'1': [],
                         '2': [],
                         '3': [],
                         '4': [],
                         '5': [],
                         '6': [],
                         '7': [],
                         }

metrics_test = {'PSNR': {j: [] for j in values},
                'SSIM': {j: [] for j in values},
                'IFC': {j: [] for j in values},
                'LPIPS': {j: [] for j in values},
                'vgg_pre': {j: [] for j in values},
                'vgg_tuned': {j: [] for j in values},
                }

metrics_test_p1 = {'PSNR': {j: [] for j in values},
                   'SSIM': {j: [] for j in values},
                   'IFC': {j: [] for j in values},
                   'LPIPS': {j: [] for j in values},
                   'vgg_pre': {j: [] for j in values},
                   'vgg_tuned': {j: [] for j in values},
                   }

features = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
glcm_features = {'LDCT': {feature: [] for feature in features},
                 'HDCT': {feature: [] for feature in features},
                 'cycleGAN': {feature: [] for feature in features},
                 }

haralick_features = {'1_0': [], '1_45': [], '1_90': [], '1_135': [], '3_0': [], '3_45': [], '3_90': [], '3_135': [],
                     '5_0': [],
                     '5_45': [], '5_90': [], '5_135': [], '1_avg': [], '3_avg': [], '5_avg': [], 'avg': []}

cov_weights = {'w_1': [],
               'w_2': [],
               'w_3': [],
               'w_4': [],
               'w_5': [],
               }


# CHECK IF A CERTAIN DIRECTORY EXISTS OTHERWISE CREATE IT
def check_directory(experiment):
    if not os.path.exists(f"../../data/test_output/mayo_testp1_exp_{experiment}/LDCT"):
        os.makedirs(f"../../data/test_output/mayo_testp1_exp_{experiment}/LDCT")
    if not os.path.exists(f"../../data/test_output/mayo_testp1_exp_{experiment}/HDCT"):
        os.makedirs(f"../../data/test_output/mayo_testp1_exp_{experiment}/HDCT")
    if not os.path.exists(f"../../data/test_output/mayo_testp1_exp_{experiment}/cycleGAN_output"):
        os.makedirs(f"../../data/test_output/mayo_testp1_exp_{experiment}/cycleGAN_output")
    if not os.path.exists(f'../../reports/metrics_te_exp_{experiment}'):
        os.makedirs(f"../../reports/metrics_te_exp_{experiment}")
    if not os.path.exists(f'../../reports/exp_{experiment}'):
        os.makedirs(f"../../reports/exp_{experiment}")
    if not os.path.exists(f'../../reports/plots/exp_{experiment}'):
        os.makedirs(f"../../reports/plots/exp_{experiment}")


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# PARAMETERS TRANSFER LEARNING VGG-16

NUM_CLASSES = 2
LR_TRANSFER = 1e-3
BATCH_SIZE_TRANSFER = 1
EPOCHS_TRANSFER = 100
annotations_transfer = '../data/csv_files/dataset_transfer_segmentation.csv'
check_point_vgg = '../../models/models_vgg/model1-1e-4_256_extended_.pth.tar'

loss_vgg = {'loss': []}

if __name__ == '__main__':
      time = open('../../reports/metrics/time.json')
      time = json.load(time)

      for i in time.keys():
          print(f'{i} avg: {np.mean(time[i])},std:  {np.std(time[i])}')