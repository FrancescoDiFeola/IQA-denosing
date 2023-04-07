from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import src.utils.util_general as config
from sklearn.preprocessing import RobustScaler
import json
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# COMPUTE THE RMSE (ROOT-MEAN-SQUARED ERROR) BETWEEN TWO INPUTS
def rmse(y_true, y_pred):
    """
    Root Mean Square Error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# COMPUTE THE NORMALIZED-RMSE (root-mean-squared-error) BETWEEN TWO INPUTS
def nrmse(y_true, y_pred):
    """
    Normalized Root Mean Square Error.
    Args:
        y_true ([np.array]): test samples
        y_pred ([np.array]): predicted samples
    Returns:
        [float]: normalized root mean square error
    """
    return rmse(y_true, y_pred) / (y_true.max() - y_true.min())


# train and test a regression tree in k-fold cross validation
def regression_tree(reg_label):
    data = pd.read_csv(f"../../reports/metrics/y_{reg_label}_.csv", index_col=0)

    X = data.loc[:, data.columns != reg_label]

    # print(X.columns)
    cols = X.columns
    y = data.loc[:, reg_label]

    # Initialize a decision tree regressor
    dt_reg = DecisionTreeRegressor(random_state=42)

    # Perform KFold cross-validation to estimate the model's performance
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = []
    feature_importance = []
    scaler = RobustScaler()
    # scaler = StandardScaler()
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        X_train = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(X_train, columns=[cols])
        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=[
            cols])  # ['FID', 'KID', 'IS', 'SNR_deno', 'Brisque', 'FD_deno', 'PaQ-2-PiQ', 'NIQE']
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        dt_reg.fit(X_train, y_train)

        # Extract the feature importances
        importances = dt_reg.feature_importances_
        feature_scores = pd.Series(dt_reg.feature_importances_, index=X_train.columns)
        feature_importance.append(dt_reg.feature_importances_)
        y_pred = dt_reg.predict(X_test)

        mse_scores.append(nrmse(y_test, y_pred))

    # Print the mean and standard deviation of the KFold cross-validation scores
    print("#########################################")
    print(f"Mean squared error y_{reg_label}: ", np.mean(mse_scores))
    print(f"Standard deviation y_{reg_label}: ", np.std(mse_scores))

    feature_scores = pd.DataFrame(feature_importance, columns=X_train.columns)
    feature_statistics = feature_scores.describe()
    # print(f"Feature importance y_{reg_label}:\n {feature_scores}")
    # print(f"Statistics on feature importance y_{reg_label}: \n {feature_scores.describe()}")
    print("#########################################")

    # feature_statistics.to_csv(f"../../reports/metrics/regression_stats_v01_{reg_label}.csv")


# it takes the regression labels, a color legend for the features and a dictionary for anular crowns. Then
# it associates each feature to a specific color respecting the feature importance order (mean value) taken from a regression_stat file
# previously computed.
def color_labels(regression_labels, features_color_legend, features_anular_circles):
    for i in regression_labels:
        data = pd.read_csv(f"../../reports/metrics/regression_stats_v01_{i}.csv", index_col=0)
        data = data.sort_values(by='mean', axis=1, ascending=False)

        for idx, j in enumerate(data.columns):
            features_anular_circles[str(idx + 1)].append(features_color_legend[j])
            print(features_anular_circles)
    color_labels = json.dumps(features_anular_circles, indent=6)
    with open(f"../../reports/metrics/color_labels_unpaired_features.json", 'w') as f:
        f.write(color_labels)


def nested_pie(regression_labels, slice_size, num_crowns, crown_width, color_legend,
               paired_tag='paired'):  # paired tag is reffered to the type of features
    sizes = []
    for i in range(0, len(regression_labels)):
        sizes.append(slice_size)

    colors_labels = open(f'../../reports/metrics/color_labels_{paired_tag}_features.json')
    colors_labels = json.load(colors_labels)

    fig, ax = plt.subplots()
    radius = 1
    for i in range(1, num_crowns + 1):
        colors = colors_labels[str(i)]

        if i == 1:
            ax.pie(sizes, colors=colors, radius=radius, labels=regression_labels,
                   wedgeprops=dict(width=crown_width, edgecolor='w'), textprops={'fontsize': 20})
        else:
            ax.pie(sizes, colors=colors, radius=radius,
                   wedgeprops=dict(width=crown_width, edgecolor='w'), textprops={'fontsize': 20})
        radius = radius - crown_width

    patch_list = []
    for j in range(0, num_crowns):
        patch = mpatches.Patch(color=list(color_legend.values())[j], label=list(color_legend.keys())[j])
        patch_list.append(patch)

    plt.legend(handles=patch_list, prop={'size': 20})
    plt.show()


if __name__ == "__main__":

    # paired_labels = ['PSNR', 'MSE', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex',
    #                 'LPIPS_squeeze']  # , 'MSE', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze'
    # unpaired_labels = ['FID', 'KID', 'IS', 'SNR_deno', 'Brisque', 'FD_deno', 'PaQ-2-PiQ', 'NIQE']
    # for i in unpaired_labels:
    #      regression_tree(i)

    # color_labels(['PSNR', 'MSE', 'SSIM', 'VIF', 'LPIPS_vgg', 'LPIPS_alex', 'LPIPS_squeeze'], config.color_legend_unpaired_features, config.anular_circles_paired_labels)
    # nested_pie(['FID', 'KID', 'IS', 'SNR', 'BRISQUE', 'RAPS-FD', 'PaQ-2-PiQ', 'NIQE'], 45, 7, 0.1, config.color_legend_paired_features, 'paired')
    nested_pie(['PSNR \n(0.07\u00B10.02)', 'MSE \n(0.07\u00B10.06)', 'SSIM \n(0.07\u00B10.02)', 'VIF \n(0.05\u00B10.01)', 'LPIPS-1 \n(0.08\u00B10.01)', 'LPIPS-2 \n(0.07\u00B10.01)', 'LPIPS-3 \n(0.07\u00B10.02)'], 45, 8, 0.1, config.color_legend_unpaired_features, 'unpaired')
