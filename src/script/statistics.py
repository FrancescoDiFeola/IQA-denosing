import matplotlib.pyplot as plt
from scipy import stats
from src.utils.util_data import *
import seaborn as sns
import json
import numpy as np


def scatter_plot(x, y, labels):
    plt.scatter(x, y, c="blue")
    linear_model = np.polyfit(x, y, 1)
    linear_model_fn = np.poly1d(linear_model)
    x_s = np.arange(round(min(x)), round(max(x)))

    plt.plot(x_s, linear_model_fn(x_s), color="red")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    pearson_coefficent = stats.pearsonr(x, y)
    spearman_coefficent = stats.spearmanr(x, y)
    plt.title(
        f'Pearson: {pearson_coefficent[0]}, p-value: {pearson_coefficent[1]} \n Spearman:{spearman_coefficent[0]}, p-value: {spearman_coefficent[1]}',
        fontsize=10)
    plt.show()


def plot_metrics(metric_name, metrics, dim1, dim2, metrics_tag):
    # metrics_unpaired = ['SNR_deno', 'Brisque', 'FD_deno', 'PaQ-2-PiQ', 'NIQE']
    # metrics_unpaired = ['MSE','PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze', 'FID', 'KID']
    set_1 = metrics[metric_name]
    for idx, unpaired in enumerate(metrics_tag):

        set_2 = metrics[unpaired]
        plt.subplot(dim1, dim2, idx + 1)
        if idx == 0:
            plt.ylabel(metric_name)
        plt.xlabel(unpaired)

        '''linear_model = np.polyfit(set_2, set_1, 1)
        linear_model_fn = np.poly1d(linear_model)
        x_s = np.arange(round(min(set_2)), round(max(set_2)) + 2)'''
        plt.scatter(set_2, set_1, c="blue", linewidths=0.2)

        # plt.plot(x_s, linear_model_fn(x_s), color="red")
        '''pearson_coefficent = stats.pearsonr(set_1, set_2)
        spearman_coefficent = stats.spearmanr(set_1, set_2)
        plt.title(
            f"Rho: {round(pearson_coefficent[0], 4)}, p-value: {round(pearson_coefficent[1], 4)} \n sigma:{round(spearman_coefficent[0], 4)}, p-value: {round(spearman_coefficent[1], 4)}",
            fontsize=10)'''


    plt.show()


def correlation_matrix(metrics, metrics_set_1, metrics_set_2, set_1, set_2, module=True, avg = True, corr_type="pearson"):
    dataframe_metrics_1 = pd.DataFrame(metrics,
                                     columns=metrics_set_1)

    dataframe_metrics_2 = pd.DataFrame(metrics,
                                     columns=metrics_set_2)

    matrix = pd.concat([dataframe_metrics_1, dataframe_metrics_2], axis=1, keys=['df1', 'df2']).corr(method=corr_type).loc['df1', 'df2']
    # matrix = dataframe_metrics.corr(method=corr_type)

    colors = sns.color_palette("Blues", as_cmap=True)
    sns.set(font_scale=1.4)
    if module == True:
        sns.heatmap(abs(matrix), vmin=abs(matrix).min(axis=1).min(), vmax=abs(matrix).max(axis=1).max(), cmap=colors, linewidths=0.8, annot=True)
        if avg == True:
            print(f"{abs(matrix).loc[set_1, set_2].to_numpy().mean()}, {abs(matrix).loc[set_1, set_2].to_numpy().std()}")
    else:
        sns.heatmap(matrix, vmin=matrix.min(axis=1).min(), vmax=matrix.max(axis=1).max(), cmap=colors,
                    linewidths=0.8, annot=True)

    plt.xticks(rotation=50)
    plt.show()



if __name__ == '__main__':

    '''['MSE', 'PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze',
     'FID', 'KID', 'IS', 'SNR_deno', 'Brisque', 'FD_deno', 'PaQ-2-PiQ',
     'NIQE']'''
    '''metrics = open('../../reports/metrics/metrics.json')
    metrics = json.load(metrics)
    raps = open('../../reports/metrics/raps.json')
    raps = json.load(raps)'''

    # patients = ['C030', 'C120', 'C124', 'C162', 'C170', 'C052', 'C067', 'C130', 'C166', 'C135']
    # metrics_paired = ['MSE', 'PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze', 'FID', 'KID', 'IS']
    # metrics_unpaired = ['SNR', 'BRISQUE', 'RAPS-FD', 'PaQ-2-PiQ', 'NIQE']
    # set_1 = metrics['SNR']['C030']['slices']

    '''metrics_all_images = {
        'MSE': [],
        'PSNR': [],
        'SSIM': [],
        'VIF': [],
        'LPIPS_vgg': [],
        'LPIPS_alex': [],
        'LPIPS_squeeze': [],
        'FID': [],
        'KID': [],
        'IS': [],
        'SNR_deno': [],
        'Brisque': [],
        'FD_deno_avg': [],
        'PaQ-2-PiQ': [],
        'NIQE': [],
    }'''

    '''for i in metrics_all_images.keys():

            try:
                for pat in patients:
                    for j in metrics[i][pat]['avg']:
                        metrics_all_images[i].append(j)

            except:
                for pat in patients:
                    for v in raps[i][pat]:
                        metrics_all_images[i].append(v)
            

    m = json.dumps(metrics_all_images, indent=6)
    with open(f"../../reports/metrics/patient_avg.json", 'w') as f:
        f.write(m)'''

    '''print(stats.pearsonr(set_1, set_2)[0])
    scatter_plot(set_2, set_1, ['SNR', 'RAPS'])'''
    metrics = open('../../reports/metrics/patient_avg.json')
    metrics = json.load(metrics)

    '''metrics_paired = ['MSE', 'PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze', 'FID', 'KID', 'IS']
    metrics_unpaired = ['SNR_deno', 'Brisque', 'FD_deno', 'PaQ-2-PiQ', 'NIQE']
    metrics_total = ['MSE', 'PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze', 'FID', 'KID', 'IS',
                     'SNR_deno', 'Brisque', 'FD_deno', 'PaQ-2-PiQ', 'NIQE']'''

    metrics_1 =  ['MSE', 'PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze']
    metrics_2 =  ['MSE', 'PSNR', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze']

    # plot_metrics('KID', metrics, 1, 5, ['SNR_deno', 'Brisque', 'FD_deno_avg', 'PaQ-2-PiQ', 'NIQE'])
    correlation_matrix(metrics, metrics_1, metrics_2, ['VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze'],['VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze'],  module=True, avg=True, corr_type="pearson")
    '''for metric in metrics_total:
        for i in metrics[metric][0]:
            metrics_all_images[metric].append(i)
    m = json.dumps(metrics_all_images, indent=6)
    with open(f"../../reports/metrics/all_images_metrics_m.json", 'w') as f:
        f.write(m)'''


