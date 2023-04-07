from math import log10

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from scipy import fftpack
from sewar.full_ref import vifp
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import Dataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips

import src.utils.util_general as config

_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from piq import brisque
import pyiqa
from frechetdist import frdist
import time

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# DATASET CLASS FOR THE VGG NETWORK
class Dataset_VGG(Dataset):
    def __init__(self, annotations, centre, width):
        super().__init__()
        self.annotations = pd.read_csv(annotations)
        self.length_dataset = len(self.annotations)
        self.window_centre = centre
        self.window_width = width

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_path = self.annotations['path_slice'].iloc[index % self.length_dataset]

        # bounding box coordinates
        x1, x2, y1, y2 = get_bounding(self.annotations, index, self.length_dataset)

        image_dicom = pydicom.dcmread(image_path)
        image = transforms(image_dicom, self.window_centre, self.window_width, x1, x2, y1, y2)

        if self.annotations['domain'].iloc[index % self.length_dataset] == 'LD':
            label = torch.Tensor([0, 1])
        elif self.annotations['domain'].iloc[index % self.length_dataset] == 'HD':
            label = torch.Tensor([1, 0])

        return image.float(), label


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# DATASET CLASS FOR THE CYCLEGAN MODEL
class Dataset(Dataset):
    def __init__(self, annotations, centre, width, segmentation):
        annotations_df = pd.read_csv(annotations)
        self.segmentation = segmentation
        if self.segmentation:
            annotations_df = annotations_df.drop(
                annotations_df[annotations_df['lung_percentage'] <= 5].index).reset_index(drop=True)
        self.annotations_ld = annotations_df.loc[annotations_df['domain'] == 'LD'].reset_index(drop=True)
        self.annotations_hd = annotations_df.loc[annotations_df['domain'] == 'HD'].reset_index(drop=True)
        #############
        # self.annotations_ld = self.annotations_ld.iloc[self.max_lung_indexes(self.annotations_ld)]
        # self.annotations_hd = self.annotations_hd.iloc[self.max_lung_indexes(self.annotations_hd)]
        #############

        # list of all the image files
        # in this case there are no pairs and the length of the dataset are not equal. We cannot use a unique index.
        self.length_dataset = max(len(self.annotations_ld), len(self.annotations_hd))
        self.low_dose_len = len(self.annotations_ld)
        self.window_centre = centre
        self.window_width = width
        print(self.low_dose_len)
        self.high_dose_len = len(self.annotations_hd)
        print(self.high_dose_len)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        low_dose_path = self.annotations_ld['path_slice'].iloc[index % self.low_dose_len]
        high_dose_path = self.annotations_hd['path_slice'].iloc[index % self.high_dose_len]
        patient_id = self.annotations_ld['patient'].iloc[index % self.low_dose_len]
        ld_dicom = pydicom.dcmread(low_dose_path)
        hd_dicom = pydicom.dcmread(high_dose_path)

        # bounding box coordinates
        if self.segmentation:
            x1_ld, x2_ld, y1_ld, y2_ld = get_bounding(self.annotations_ld, index, self.low_dose_len)
            x1_hd, x2_hd, y1_hd, y2_hd = get_bounding(self.annotations_hd, index, self.high_dose_len)
            low_dose_img = transforms(ld_dicom, self.window_centre, self.window_width, x1_ld, x2_ld, y1_ld, y2_ld)
            high_dose_img = transforms(hd_dicom, self.window_centre, self.window_width, x1_hd, x2_hd, y1_hd, y2_hd)
        else:
            low_dose_img = transforms_no_bounding(ld_dicom, self.window_centre, self.window_width)
            high_dose_img = transforms_no_bounding(hd_dicom, self.window_centre, self.window_width)

        return low_dose_img.float(), high_dose_img.float(), patient_id

    @staticmethod
    def max_lung_indexes(annotation):
        patients = annotation['patient'].unique()
        max_lung_percentage = [max(annotation['lung_percentage'].loc[annotation['patient'] == i]) for i in
                               patients]

        indexes = [int(k) for (i, j) in zip(patients, max_lung_percentage) for k in
                   list(np.where((annotation['patient'] == i) & (annotation['lung_percentage'] == j)))]
        return indexes


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
# DATASET CLASS FOR SCAPIS
class ScapisDataset(Dataset):
    def __init__(self, annotations, centre, width, segmentation):
        annotations_df = pd.read_csv(annotations)
        self.segmentation = segmentation
        if self.segmentation:
            annotations_df = annotations_df.drop(
                annotations_df[annotations_df['lung_percentage'] <= 5].index).reset_index(drop=True)
        self.annotations = annotations_df.iloc[self.max_lung_indexes(
            annotations_df)]  # for each patient we take the slice with the highest lung percentage
        # list of all the image files
        # in this case there are no pairs and the length of the dataset are not equal. We cannot use a unique index.
        self.length_dataset = len(self.annotations)
        self.window_centre = centre
        self.window_width = width

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        path = self.annotations['path_slice'].iloc[index % self.length_dataset]
        patient_id = self.annotations['patient'].iloc[index % self.length_dataset]
        dicom = pydicom.dcmread(path)

        # bounding box coordinates
        if self.segmentation:
            x1, x2, y1, y2 = get_bounding(self.annotations, index, self.length_dataset)
            img = transforms(dicom, self.window_centre, self.window_width, x1, x2, y1, y2)
        else:
            img = transforms_no_bounding(dicom, self.window_centre, self.window_width)

        return img.float(), patient_id, x1

    @staticmethod
    def max_lung_indexes(annotation):
        patients = annotation['patient'].unique()
        max_lung_percentage = [max(annotation['lung_percentage'].loc[annotation['patient'] == i]) for i in
                               patients]

        indexes = [int(k) for (i, j) in zip(patients, max_lung_percentage) for k in
                   list(np.where((annotation['patient'] == i) & (annotation['lung_percentage'] == j)))]
        return indexes


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
def air_masking(dicom):
    data_hu = convert_in_hu(dicom)
    resized_image = cv2.resize(data_hu, (256, 256))
    return np.array(np.where(resized_image > -600, False, True), dtype=np.int16)


# GET THE BOUNDING BOX COORDINATES
def get_bounding(annotation, idx, lenght):
    bounding = []
    bounding.append(int(annotation['x1_selected'].iloc[idx % lenght]))
    bounding.append(int(annotation['x2_selected'].iloc[idx % lenght]))
    bounding.append(int(annotation['y1_selected'].iloc[idx % lenght]))
    bounding.append(int(annotation['y2_selected'].iloc[idx % lenght]))

    return bounding


# APPLY ALL THE TRANFORMATIONS TO THE IMAGE (NO BOUNDING)
def transforms_no_bounding(dicom, window_centre, window_width, tensor_output=True):
    data_hu = convert_in_hu(dicom)
    windowed_img = window_image(data_hu, window_centre, window_width)
    # print(cropped_image.shape)
    normalized_img = normalization(windowed_img)
    resized_image = cv2.resize(normalized_img, (256, 256))  # np.resize da problemi
    # print(resized_image.shape)
    if tensor_output:
        img_shaped = change_shape(resized_image)
        # print(img_rgb.shape)
        tensor_image = (torch.from_numpy(img_shaped))
        # print(tensor_image.shape)
        return tensor_image
    else:
        return resized_image.astype('float32')


# APPLY ALL THE TRANFORMATIONS TO THE IMAGE (BOUNDING)
def transforms(dicom, window_centre, window_width, x1, x2, y1, y2, tensor_output=True):
    data_hu = convert_in_hu(dicom)
    windowed_img = window_image(data_hu, window_centre, window_width)
    cropped_image = crop(windowed_img, x1, x2, y1, y2)
    # print(cropped_image.shape)
    normalized_img = normalization(cropped_image)
    resized_image = cv2.resize(normalized_img, (256, 256))  # np.resize da problemi
    # print(resized_image.shape)
    if tensor_output:
        img_shaped = change_shape(resized_image)
        # print(img_rgb.shape)
        tensor_image = (torch.from_numpy(img_shaped))
        # print(tensor_image.shape)
        return tensor_image
    else:
        return resized_image.astype('float32')


# APPLY THE LINEAR TRANSFORMATION TO GET THE HU VALUES (y =m*x+q)
def convert_in_hu(dicom_file):
    image = dicom_file.pixel_array
    intercept = dicom_file.RescaleIntercept
    slope = dicom_file.RescaleSlope
    image = slope * image + intercept
    return image


# IMAGE WINDOWING GIVEN A CENTRE AND THE WINDOW WIDTH
def window_image(hu_img, window_center, window_width):
    img_w = hu_img.copy()
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img_w[img_w < img_min] = img_min
    img_w[img_w > img_max] = img_max

    return img_w


# CROP THE IMAGE USING THE BOUNDIG BOX COORDINATES
def crop(img, x1, x2, y1, y2):
    return img[y1:y2, x1:x2]


# NORMALIZATION IN THE RANGE [-1, 1] or [0,1]
def normalization(img, range='-11'):
    normalized_input = (img - np.min(img)) / (np.max(img) - np.min(img))
    if range == '01':
        return normalized_input
    else:
        return 2 * normalized_input - 1


def change_shape(image):
    image_shaped = np.empty((1, image.shape[0], image.shape[1]), dtype=image.dtype)
    image_shaped[:, :, :] = image[np.newaxis, :, :].astype(np.float32)
    return image_shaped


# CONVERT A 1-CHANNEL INPUT TO A 3-CHANNEL OUTPUT
def change_shape_3channels(image):
    image_3channel = torch.empty((image.shape[0], 3, image.shape[2], image.shape[3]), dtype=torch.float32)
    for i in range(0, image.shape[0], 1):
        im = image[i, :, :, :]
        image_1 = torch.cat([im, im, im], dim=0)
        image_3channel[i, :, :, :] = image_1

    return image_3channel


# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# COMPUTE PIXELWISE METRICS
def compute_metrics(img, original, id):
    tic_1 = time.time()
    ssim_score = float(structural_similarity(original, img))
    toc_1 = time.time()
    tic_2 = time.time()
    psnr_score = float(psnr(original, img))
    toc_2 = time.time()
    tic_3 = time.time()
    mse_score = float(mean_squared_error(original, img))
    toc_3 = time.time()
    print(f'SSIM: {toc_1 - tic_1}, PSNR: {toc_2 - tic_2}, MSE: {toc_3 - tic_3}')
    config.metrics['PSNR'][id]['slices'].append(psnr_score)
    config.metrics['SSIM'][id]['slices'].append(ssim_score)
    config.metrics['MSE'][id]['slices'].append(mse_score)


# COMPUTE HARALICK LOSS
def compute_haralick(img1, img2):
    contrast_1, dissimilarity_1, homogeneity_1, ASM_1, energy_1, correlation_1 = glcm_feature(img1)
    contrast_2, dissimilarity_2, homogeneity_2, ASM_2, energy_2, correlation_2 = glcm_feature(img2)
    # haralick loss terms
    loss_h1 = abs(contrast_1 - contrast_2)  # 10e-3
    loss_h2 = abs(dissimilarity_1 - dissimilarity_2)
    loss_h3 = abs(homogeneity_1 - homogeneity_2)
    loss_h4 = abs(ASM_1 - ASM_2)
    loss_h5 = abs(energy_1 - energy_2)
    loss_h6 = abs(correlation_1 - correlation_2)
    print(loss_h1, loss_h2, loss_h3, loss_h4, loss_h5, loss_h6)
    return loss_h1, loss_h2, loss_h3, loss_h4, loss_h5


# COMPUTE THE AZIMUTHALLY AVERAGED RADIAL PRIFILE
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).
    """
    # compute the 2D discrete transform
    F_img = fftpack.fftshift(fftpack.fft2(image))
    # power spectral density
    psd2D_img = np.abs(F_img) ** 2

    # Calculate the indices from the image (y: rows, x: columns)
    y, x = np.indices(psd2D_img.shape)

    # calculate the center of the image
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])

    # calculate the hypotenusa for each position respect to the center
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)  # .flat is a 1-D iterator over the array.
    r_sorted = r.flat[ind]
    i_sorted = psd2D_img.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # consider the location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


# GIVEN TWO LISTS COMPUTE THE FRECHET DISTANCE
def f_distance(list1, list2):
    l = [[a, log10(b)] for a, b in zip(range(0, len(list1)), list1)]
    print(l)
    d = [[a, log10(b)] for a, b in zip(range(0, len(list1)), list2)]
    print(d)
    return frdist(l, d)


def get_annotation(name):
    if name == 'test_mayo':
        test_annotation = pd.read_csv(config.annotations_test_mayo_p1)
        annotation_1 = test_annotation.loc[test_annotation['domain'] == 'LD'].reset_index(drop=True)
        annotation_2 = test_annotation.loc[test_annotation['domain'] == 'HD'].reset_index(drop=True)
    return annotation_1, annotation_2


# COMPUTE HARALICK'S FEATURES FROM GRAY-LEVEL-COOCCURENCE MATRIX (GLCM)
def glcm_feature(img):  # domain
    img_8bit = conv_to_8bit(img)
    '''glcm_1_0 = graycomatrix(img_8bit, distances=[1], angles=[0], levels=256)  # 256
    glcm_1_45 = graycomatrix(img_8bit, distances=[1], angles=[45], levels=256)
    glcm_1_90 = graycomatrix(img_8bit, distances=[1], angles=[90], levels=256)
    glcm_1_135 = graycomatrix(img_8bit, distances=[1], angles=[135], levels=256)
    glcm_1_avg = (glcm_1_0 + glcm_1_45 + glcm_1_90 + glcm_1_135) / 4'''
    '''plt.imshow(glcm_1_avg[:, :, 0, 0])
    plt.colorbar()
    plt.show()'''
    '''glcm_3_0 = graycomatrix(img_8bit, distances=[3], angles=[0], levels=256)  # 256
    glcm_3_45 = graycomatrix(img_8bit, distances=[3], angles=[45], levels=256)
    glcm_3_90 = graycomatrix(img_8bit, distances=[3], angles=[90], levels=256)
    glcm_3_135 = graycomatrix(img_8bit, distances=[3], angles=[135], levels=256)
    glcm_3_avg = (glcm_3_0 + glcm_3_45 + glcm_3_90 + glcm_3_135) / 4'''

    glcm_5_0 = graycomatrix(img_8bit, distances=[5], angles=[0], levels=256)  # 256
    glcm_5_45 = graycomatrix(img_8bit, distances=[5], angles=[45], levels=256)
    glcm_5_90 = graycomatrix(img_8bit, distances=[5], angles=[90], levels=256)
    glcm_5_135 = graycomatrix(img_8bit, distances=[5], angles=[135], levels=256)
    glcm_5_avg = (glcm_5_0 + glcm_5_45 + glcm_5_90) / 3
    glcm_5_avg_2 = (glcm_5_0 + glcm_5_45 + glcm_5_90 + glcm_5_135) / 4
    # glcm_avg = (glcm_1_avg + glcm_3_avg + glcm_5_avg) / 3
    contrast_2 = graycoprops(glcm_5_avg_2, "contrast")[0][0]
    contrast = graycoprops(glcm_5_avg, "contrast")[0][0]
    dissimilarity = graycoprops(glcm_5_avg, "dissimilarity")[0][0]
    homogeneity = graycoprops(glcm_5_avg, "homogeneity")[0][0]
    ASM = graycoprops(glcm_5_avg, "ASM")[0][0]
    energy = graycoprops(glcm_5_avg, "energy")[0][0]
    correlation = graycoprops(glcm_5_avg, "correlation")[0][0]

    '''config.glcm_features[domain]['contrast'].append(contrast)
    config.glcm_features[domain]['dissimilarity'].append(dissimilarity)
    config.glcm_features[domain]['homogeneity'].append(homogeneity)
    config.glcm_features[domain]['ASM'].append(ASM)
    config.glcm_features[domain]['energy'].append(energy)
    config.glcm_features[domain]['correlation'].append(correlation)'''

    '''config.haralick_features['1_0'].append(graycoprops(glcm_1_0, "energy")[0][0])
    config.haralick_features['1_45'].append(graycoprops(glcm_1_45, "energy")[0][0])
    config.haralick_features['1_90'].append(graycoprops(glcm_1_90, "energy")[0][0])
    config.haralick_features['1_135'].append(graycoprops(glcm_1_135, "energy")[0][0])
    config.haralick_features['3_0'].append(graycoprops(glcm_3_0, "energy")[0][0])
    config.haralick_features['3_45'].append(graycoprops(glcm_3_45, "energy")[0][0])
    config.haralick_features['3_90'].append(graycoprops(glcm_3_90, "energy")[0][0])
    config.haralick_features['3_135'].append(graycoprops(glcm_3_135, "energy")[0][0])
    config.haralick_features['5_0'].append(graycoprops(glcm_5_0, "energy")[0][0])
    config.haralick_features['5_45'].append(graycoprops(glcm_5_45, "energy")[0][0])
    config.haralick_features['5_90'].append(graycoprops(glcm_5_90, "energy")[0][0])
    config.haralick_features['5_135'].append(graycoprops(glcm_5_135, "energy")[0][0])
    config.haralick_features['1_avg'].append(graycoprops(glcm_1_avg, "energy")[0][0])
    config.haralick_features['3_avg'].append(graycoprops(glcm_3_avg, "energy")[0][0])
    config.haralick_features['5_avg'].append(graycoprops(glcm_5_avg, "energy")[0][0])
    config.haralick_features['avg'].append(graycoprops(glcm_avg, "energy")[0][0])'''
    print(contrast_2)
    print(contrast)
    return contrast, dissimilarity, homogeneity, ASM, energy, correlation
    # return glcm_1_0, glcm_1_45, glcm_1_90, glcm_1_135, glcm_3_0, glcm_3_45, glcm_3_90, glcm_3_135, glcm_5_0, glcm_5_45, glcm_5_90, glcm_5_135


# COMPUTE THE PEAK SIGNAL-TO-NOISE RATIO (PSNR) BETWEEN TWO IMAGES
def psnr(original, compressed):
    mse = mean_squared_error(original, compressed)
    if mse == 0:  # MSE is zero means no noise is present in the signal. Therefore, PSNR have no importance.
        return 100

    max_pixel = 1  # 4095
    psnr = 10 * log10((max_pixel ** 2) / mse)  # 20 * log10(max_pixel / sqrt(mse))
    return psnr


# COMPUTE THE MSE BETWEEN TWO IMAGES
def mean_squared_error(original, compressed):
    mse = np.square(np.subtract(original, compressed)).mean()
    return mse


# CONVERT AN INPUT IN AN 8-BIT OUTPUT
def conv_to_8bit(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image


# FIND THE MINIMUM AND MAXIMUM VALUE IN AN IMAGE
def min_max(data):
    row_max = list()
    for i in range(0, 256):
        row_max.append(max(data[i, :]))

    row_min = list()
    for i in range(0, 256):
        row_min.append(min(data[i, :]))

    return min(row_min), max(row_max)


# RETURNS A LIST OF BOUNDING BOXES TO EXTRACT PATCHES FROM AN IMAGE
def calculate_slice_bboxes(
        image_height: int,
        image_width: int,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
):
    """
        Given the height and width of an image, calculates how to divide the image into
        overlapping slices according to the height and width provided. These slices are returned
        as bounding boxes in xyxy format.
        :param image_height: Height of the original image.
        :param image_width: Width of the original image.
        :param slice_height: Height of each slice
        :param slice_width: Width of each slice
        :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
        :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
        :return: a list of bounding boxes in xyxy format
        """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def extract_patches(patch_x, patch_y, patches, hd, ld):

    hd_patches = torch.empty((len(patches), 3, patch_x, patch_y), dtype=torch.uint8)

    for slice_idx, slice_bbox in sorted(enumerate(patches)):
        hd_patch = conv_to_8bit(hd[slice_bbox[1]:slice_bbox[3], slice_bbox[0]:slice_bbox[2]].copy())

        hd_patch = change_shape_3channels(torch.Tensor(np.reshape(hd_patch, (1, 1, patch_x, patch_y))))
        hd_patches[slice_idx, :, :, :] = hd_patch

    ld_patches = torch.empty((len(patches), 3, patch_x, patch_y), dtype=torch.uint8)

    for slice_idx, slice_bbox in sorted(enumerate(patches)):
        ld_patch = conv_to_8bit(ld[slice_bbox[1]:slice_bbox[3], slice_bbox[0]:slice_bbox[2]].copy())

        ld_patch = change_shape_3channels(torch.Tensor(np.reshape(ld_patch, (1, 1, patch_x, patch_y))))
        ld_patches[slice_idx, :, :, :] = ld_patch

    return hd_patches, ld_patches


def plot_img(x, pname):
    plt.imshow(x, cmap='gray')
    plt.title(pname)
    plt.axis('off')
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

   pass