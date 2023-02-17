import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.stats import zscore
import numpy as np
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and split into features and target
X = pd.read_csv("C:\Dropbox (BGU)\Sleep Deprivation\Functions\Data Preparation & Extraction\FeaturesOnly.csv")
velocity = pd.read_csv(r"C:\Dropbox (BGU)\Sleep Deprivation\Functions\Data Preparation & Extraction\velocity_only.csv")
y = pd.read_csv("C:\Dropbox (BGU)\Sleep Deprivation\Functions\Data Preparation & Extraction\RT_only.csv")
timeAwakeLabels = pd.read_csv(r"C:\Dropbox (BGU)\Sleep Deprivation\Functions\Data Preparation & Extraction\timeAwake_only.csv")

# Interpolate missing values
df = pd.DataFrame(X)
df['timeAwake'] = timeAwakeLabels
df['velocity'] = velocity
df['RT'] = y
df = df.replace([np.inf, -np.inf], np.nan)
total_nan = df.isna().sum().sum() # count total number of NaNs
for feature in df.columns:
    df[feature] = df.groupby('timeAwake')[feature].apply(lambda x: x.interpolate())

# check if there are any NaNs left:
X = df.drop('timeAwake', axis=1)
print('Number of NaNs left:', X.isna().sum().sum()) # check if there are any NaNs left
X_interp_zscore = zscore(X)

# Split data into training and testing sets (reaction time and velocity), 80/20 split
column_labels = X.columns
X_train, X_test, y_train, y_test, awake_train, awake_test, velo_train, velo_test = train_test_split(X_interp_zscore, y, timeAwakeLabels, velocity, test_size=0.2, random_state=42)
X_train = pd.DataFrame(X_train, columns=column_labels)
X_test = pd.DataFrame(X_test, columns=column_labels)

## Define feature coalitions: ##
# Complexity feature variables
mse_cols = X_train.filter(regex='MSE').columns
train_mse_features = pd.DataFrame(X_train[mse_cols].values)
train_mse_features.columns = mse_cols
test_mse_features = pd.DataFrame(X_test[mse_cols].values)
test_mse_features.columns = mse_cols

# Criticality feature variables
crit_cols = X_train.filter(regex='Critical').columns
train_crit_features = pd.DataFrame(X_train[crit_cols].values)
train_crit_features.columns = crit_cols
test_crit_features = pd.DataFrame(X_test[crit_cols].values)
test_crit_features.columns = crit_cols

# Powerspectrum feature variables
ps_cols = X_train.filter(regex='(Alpha|Beta|Gamma|Delta|Theta)').columns
train_PS_features = pd.DataFrame(X_train[ps_cols].values)
train_PS_features.columns = ps_cols
test_PS_features = pd.DataFrame(X_test[ps_cols].values)
test_PS_features.columns = ps_cols
test_PS_features = test_PS_features.drop(test_PS_features.columns[-1], axis=1)
train_PS_features = train_PS_features.drop(train_PS_features.columns[-1], axis=1)

# Define PCA feature variables
pca = PCA(n_components=25)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# set feature coalitions
train_feature_coalitions = [
    [train_mse_features],
    [train_crit_features],
    [train_PS_features],
    [train_PS_features, train_crit_features],
    [train_PS_features, train_mse_features],
    [train_crit_features, train_mse_features],
    [train_PS_features, train_crit_features, train_mse_features]
]
train_feature_coalitions = [pd.concat(coalition, axis=1) for coalition in train_feature_coalitions]
test_feature_coalitions = [
    [test_mse_features],
    [test_crit_features],
    [test_PS_features],
    [test_PS_features, test_crit_features],
    [test_PS_features, test_mse_features],
    [test_crit_features, test_mse_features],
    [test_PS_features, test_crit_features, test_mse_features]
]
test_feature_coalitions = [pd.concat(coalition, axis=1) for coalition in test_feature_coalitions]

# Loop through feature coalitions
for i, features in enumerate(train_feature_coalitions):
    # Select features
    # X_train_subset = train_feature_coalitions[i]
    # X_test_subset = test_feature_coalitions[i]
    # X_train_subset = X_train_pca
    # X_test_subset = X_test_pca
    X_train_subset = X_train
    X_test_subset = X_test

    ## Model fittings: ##

    # Linear Regression
    # lr = LinearRegression()
    # # Define best features from RFE
    # selector = RFE(lr, n_features_to_select=25)
    # selector = selector.fit(X_train_subset, y_train.values.ravel())
    # X_selected_train = selector.transform(X_train_subset)
    # X_selected_test = selector.transform(X_test_subset)
    # print(X_train_subset.columns[selector.support_])
    # lr.fit(X_selected_train, y_train)
    # y_pred_lr = lr.predict(X_selected_test)
    # print('Linear Regression:')
    # # print(f'Coalition: {train_feature_coalitions[i].columns}')
    # print(f'Linear Regression MAE: {mean_absolute_error(y_test, y_pred_lr)}')
    # print(f'Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr)}')
    # print(f'Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr))}')
    # print(f'Linear Regression R-Squared: {r2_score(y_test, y_pred_lr)}\n')
    # del X_selected_test, X_selected_train, selector

    # Random Forest
    # rf = RandomForestRegressor()
    # # Define best features from RFE
    # selector = RFE(rf, n_features_to_select=25)
    # # selector = selector.fit(X_train_subset, y_train.values.ravel())
    # selector = selector.fit(X_train_subset, velo_train.values.ravel())
    # X_selected_train = selector.transform(X_train_subset)
    # X_selected_test = selector.transform(X_test_subset)
    # print('Random Forest:')
    # print(X_train_subset.columns[selector.support_])
    # rf.fit(X_selected_train, velo_train.values.ravel())
    # y_pred_rf = rf.predict(X_selected_test)
    # print(f'Random Forest MAE: {mean_absolute_error(velo_test, y_pred_rf)}')
    # print(f'Random Forest MSE: {mean_squared_error(velo_test, y_pred_rf)}')
    # print(f'Random Forest RMSE: {np.sqrt(mean_squared_error(velo_test, y_pred_rf))}')
    # print(f'Random Forest R-Squared: {r2_score(velo_test, y_pred_rf)}\n')
    # del X_selected_test, X_selected_train, selector

    # LGBMRegressor for velocity
    LGBM_velo = lgb.LGBMRegressor()
    # Define best features from RFE:
    selector_velo = RFE(LGBM_velo, n_features_to_select=25)
    selector_velo = selector_velo.fit(X_train_subset, velo_train.values.ravel())
    X_selected_train_velo = selector_velo.transform(X_train_subset)
    X_selected_test_velo = selector_velo.transform(X_test_subset)
    X_selected_train_velo = pd.DataFrame(X_selected_train_velo)
    X_selected_test_velo = pd.DataFrame(X_selected_test_velo)
    X_selected_train_velo.columns = X_train_subset.columns[selector_velo.support_]
    X_selected_test_velo.columns = X_test_subset.columns[selector_velo.support_]
    print('Light GBM:')
    print(X_train_subset.columns[selector_velo.support_])
    LGBM_velo.fit(X_selected_train_velo, np.squeeze(velo_train))
    y_pred_LGBM_velo = LGBM_velo.predict(X_selected_test_velo)
    print(f'LGBM MAE: {mean_absolute_error(velo_test, y_pred_LGBM_velo)}')
    print(f'LGBM MSE: {mean_squared_error(velo_test, y_pred_LGBM_velo)}')
    print(f'LGBM RMSE: {np.sqrt(mean_squared_error(velo_test, y_pred_LGBM_velo))}')
    print(f'LGBM R-Squared: {r2_score(velo_test, y_pred_LGBM_velo)}\n')
    # del X_selected_test, X_selected_train, selector

    # LGBMRegressor for reaction time
    LGBM_RT = lgb.LGBMRegressor()
    # Define best features from RFE
    selector_RT = RFE(LGBM_RT, n_features_to_select=25)
    selector_RT = selector_RT.fit(X_train_subset, y_train.values.ravel())
    X_selected_train_RT = selector_RT.transform(X_train_subset)
    X_selected_test_RT = selector_RT.transform(X_test_subset)
    X_selected_test_RT = pd.DataFrame(X_selected_test_RT)
    X_selected_train_RT = pd.DataFrame(X_selected_train_RT)
    # preserve column names:
    X_selected_train_RT.columns = X_train_subset.columns[selector_RT.support_]
    X_selected_test_RT.columns = X_test_subset.columns[selector_RT.support_]
    print('Light GBM:')
    print(X_train_subset.columns[selector_RT.support_])
    LGBM_RT.fit(X_selected_train_RT, np.squeeze(y_train))
    y_RT_pred_LGBM = LGBM_RT.predict(X_selected_test_RT)
    print(f'LGBM MAE: {mean_absolute_error(y_test, y_RT_pred_LGBM)}')
    print(f'LGBM MSE: {mean_squared_error(y_test, y_RT_pred_LGBM)}')
    print(f'LGBM RMSE: {np.sqrt(mean_squared_error(y_test, y_RT_pred_LGBM))}')
    print(f'LGBM R-Squared: {r2_score(y_test, y_RT_pred_LGBM)}\n')
    # del X_selected_test, X_selected_train, selector

    # Kernel Ridge
    # kr = KernelRidge()
    # # Define best features from RFE
    # selector = RFE(kr, n_features_to_select=25)
    # selector = selector.fit(X_train_subset, y_train.values.ravel())
    # X_selected_train = selector.transform(X_train_subset)
    # X_selected_test = selector.transform(X_test_subset)
    # print('Kernel Ridge:')
    # print(X_train_subset.columns[selector.support_])
    # kr.fit(X_selected_train, y_train)
    # y_pred_kr = kr.predict(X_selected_test)
    # print(f'Kernel Ridge MAE: {mean_absolute_error(y_test, y_pred_kr)}')
    # print(f'Kernel Ridge MSE: {mean_squared_error(y_test, y_pred_kr)}')
    # print(f'Kernel Ridge RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_kr))}')
    # print(f'Kernel Ridge R-Squared: {r2_score(y_test, y_pred_kr)}\n')
    # del X_selected_test, X_selected_train, selector

    # ANN
    # ann = MLPRegressor()
    # # Define best features from RFE
    # selector = RFE(ann, n_features_to_select=25)
    # selector = selector.fit(X_train_subset, y_train.values.ravel())
    # X_selected_train = selector.transform(X_train_subset)
    # X_selected_test = selector.transform(X_test_subset)
    # print('ANN:')
    # print(X_train_subset.columns[selector.support_])
    # ann.fit(X_selected_train, y_train.values.ravel())
    # y_pred_ann = ann.predict(X_selected_test)
    # print(f'ANN MAE: {mean_absolute_error(y_test, y_pred_ann)}')
    # print(f'ANN MSE: {mean_squared_error(y_test, y_pred_ann)}')
    # print(f'ANN RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ann))}')
    # print(f'ANN R-Squared: {r2_score(y_test, y_pred_ann)}\n')
    # del X_selected_test, X_selected_train, selector

# load LGBM_RT model from pickle file
import pickle
with open('LGBM_RT_model.pkl', 'rb') as f:
    LGBM_RT = pickle.load(f)

# load LGBM_velo model from pickle file
with open('LGBM_velo_model.pkl', 'rb') as f:
    LGBM_velo = pickle.load(f)

## Plotting: ##
# Joint plot:
sns.set()
sns.set_style('whitegrid')
r2 = r2_score(y_test, y_RT_pred_LGBM)
plot_data = {'predictions': y_RT_pred_LGBM, 'true_labels': np.squeeze(y_test), 'Time Awake (h)': np.squeeze(awake_test)}
plt = sns.jointplot(x="predictions", y="true_labels", data=plot_data, hue="Time Awake (h)")
#plt.title("Mean RT Prediction Vs True")
minimum_pred = np.min(y_RT_pred_LGBM)
minimum_true = np.min(np.squeeze(y_test))
maximum_pred = np.max(y_RT_pred_LGBM)
maximum_true = np.max(np.squeeze(y_test))
maximum_overall = np.max([maximum_pred, maximum_true])
minimum_overall = np.min([minimum_pred, minimum_true])
sns.lineplot(x=[minimum_overall, maximum_overall], y=[minimum_overall, maximum_overall])
plt.ax_joint.set_xlim([minimum_overall-0.05*minimum_overall, maximum_overall+0.05*maximum_overall])
plt.ax_joint.set_ylim([minimum_overall-0.05*minimum_overall, maximum_overall+0.05*maximum_overall])
plt.ax_joint.text(minimum_overall+0.005*minimum_overall, maximum_overall-0.005*maximum_overall, f'R-squared = {r2:.3f}', fontsize=22)
plt.ax_joint.set_xlabel('True values (s)', fontsize=22)
plt.ax_joint.set_ylabel('Predicted values (s)', fontsize=22)
plt.ax_joint.tick_params(labelsize=22)
plt.ax_joint.legend(loc='lower right', fontsize=20, bbox_to_anchor=(1, 0))

# Plotting SHAP values:
import shap
explainer_RT = shap.Explainer(LGBM_RT, X_selected_train_RT)
explainer_velo = shap.Explainer(LGBM_velo, X_selected_train_velo)
shap_values_RT = explainer_RT(X_selected_train_RT)
shap_values_velo = explainer_velo(X_selected_train_velo)
shap.summary_plot(shap_values_RT, X_selected_train_RT, max_display=25)
shap.summary_plot(shap_values_velo, X_selected_train_velo, max_display=25)

# save the model to disk
# filename = 'LGBM_RT.sav'
# pickle.dump(LGBM_RT, open(filename, 'wb'))
# filename = 'LGBM_velo.sav'
# pickle.dump(LGBM_velo, open(filename, 'wb'))

# load the model from disk
# filename = 'LGBM_RT.sav'
# LGBM_RT = pickle.load(open(filename, 'rb'))
# filename = 'LGBM_velo.sav'
# LGBM_velo = pickle.load(open(filename, 'rb'))


# plot critical alpha to time awake
import scipy.stats as stats
import statsmodels.api as sm
x = df['RT']
y = df["'CriticalAlpha'"]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope*x+intercept
rsq = r_value**2

# Violin plot:
# fig, ax = plt.subplots()
# plt.rc('text', usetex=True)
# sns.violinplot(x='timeAwake', y="'CriticalAlpha'", data=df, ax=ax)
# plot a line that goes through the mean of each group in the Violin plot
# grouped = df.groupby('timeAwake')
# regression_lines = {}
# p_values = {}
# for name, group in grouped:
#     X = group['timeAwake'].values.reshape(-1, 1)
#     y = group["'CriticalAlpha'"]
#     model = sm.OLS(y, sm.add_constant(X)).fit()
#     regression_lines[name] = model.params[0] + model.params[1] * X
#     p_values[name] = model.pvalues_[1]
#     sns.lineplot(x=X.flatten(), y=regression_lines[name].flatten(), ax=ax, color='black')
#
# for i, timeAwake in enumerate(p_values.keys()):
#     plt.annotate("p = {:.3f}".format(p_values[timeAwake]), (timeAwake, regression_lines[timeAwake]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=16)
# plt.show()
# # plt.text(0.05, 0.95, 'R-squared = {:.3f}'.format(rsq), transform=ax.transAxes, fontsize=22)
# ax.set_title('Critical Alpha Metric over Time Awake', fontsize=22)
# ax.set_xlabel('Time Awake (h)', fontsize=22)
# ax.set_ylabel(r'Critical Alpha ($\alpha$)', fontsize=22)
# ax.xaxis.set_tick_params(labelsize=22)
# ax.yaxis.set_tick_params(labelsize=22)

# Violin Plot of Critical Alpha over Time Awake:
fig, ax = plt.subplots()
sns.violinplot(x='timeAwake', y="'CriticalAlpha'", data=df, ax=ax)
grouped = df.groupby('timeAwake')
regression_lines = {}
p_values = {}
for name, group in grouped:
    X = group['timeAwake'].values.reshape(-1, 1)
    y = group["'CriticalAlpha'"]
    model = sm.OLS(y, sm.add_constant(X)).fit()
    regression_lines[name] = model.params[0] + model.params[1] * X
    p_values[name] = model.pvalues[1]
    sns.lineplot(x=X.flatten(), y=regression_lines[name].flatten(), ax=ax, color='black')
for i, timeAwake in enumerate(p_values.keys()):
    plt.annotate("p = {:.3f}".format(p_values[timeAwake]),(timeAwake, regression_lines[timeAwake][0]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=16)
plt.show()
ax.set_title('Critical Alpha Metric over Time Awake', fontsize=22)
ax.set_xlabel('Time Awake (h)', fontsize=22)
ax.set_ylabel(r'Critical Alpha ($\alpha$)', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)

# plot critical alpha to reaction time:
from sklearn.linear_model import BayesianRidge

fig, ax = plt.subplots()
x = df['RT']
x = x.values.reshape(-1, 1)
y = df["'CriticalAlpha'"]
# y = y.values.reshape(-1, 1)
model = BayesianRidge().fit(x, y)
sns.scatterplot(x='RT', y="'CriticalAlpha'", data=df, hue='timeAwake')
plt.plot(x, model.predict(x), color='black')
plt.text(0.05, 0.95, 'Slope: {:.3f}'.format(model.coef_[0]), transform=ax.transAxes, fontsize=22)
plt.text(0.05, 0.9, 'Intercept: {:.3f}'.format(model.intercept_), transform=ax.transAxes, fontsize=22)
plt.text(0.05, 0.85, 'R-squared: {:.3f}'.format(model.score(x, y)), transform=ax.transAxes, fontsize=22)
plt.show()
ax = plt.gca()
# ax.set_title('Critical Alpha Metric over Reaction Time', fontsize=22)
leg = plt.legend()
leg.set_title('Time Awake (h)', prop={'size': 18})
ax.set_xlabel('Reaction Time (s)', fontsize=22)
ax.set_ylabel(r'Critical Alpha ($\alpha$)', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)

# plot averaged critical alpha to time awake with standard error of the mean:
fig, ax = plt.subplots()
grouped = df.groupby('timeAwake')
mean = grouped["'CriticalAlpha'"].mean()
sem = grouped["'CriticalAlpha'"].apply(lambda x: np.std(x) / np.sqrt(len(x)))
# plot the mean and SEM:
plt.errorbar(mean.index, mean.values, yerr=sem.values, fmt='o', capsize=5)
# fit a line through the mean values:
coeffs = np.polyfit(mean.index, mean.values, 1)
x = np.array([mean.index.min(), mean.index.max()])
y = np.polyval(coeffs, x)
plt.plot(x, y, '-', color='black')
# calculate the r-squared value:
# residuals = mean.values - np.polyval(coeffs, mean.index)
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((mean.values - np.mean(mean.values))**2)
# r_squared = 1 - ss_res  / ss_tot
# add the slope, intercept and r-squared values to the plot:
# text = f'slope = {coeffs[0]:.2f}, intercept = {coeffs[1]:.2f}\nR^2 = {r_squared:.2f}'
# plt.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=22, verticalalignment='top')
ax.set_xlabel('Time Awake (h)', fontsize=22)
ax.set_ylabel(r'Critical Alpha ($\alpha$)', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)


# plot critical alpha to velocity
fig, ax = plt.subplots()
sns.lmplot(x='velocity', y="'CriticalAlpha'", data=df, hue='timeAwake', fit_reg=False)
plt.show()
ax = plt.gca()
# get linear fit of the data:
x = df['velocity']
y = df["'CriticalAlpha'"]
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
plt.plot(x[:, 1], results.predict(), '-', color='black')
# add p-values to the plot:
plt.title('P-value = {:.4f}'.format(results.pvalues[1]), fontsize=18)
leg = plt.legend(loc='upper right')
leg.set_title('Time Awake (h)', prop={'size': 18})
ax.set_xlabel('Velocity (1/s)', fontsize=22)
ax.set_ylabel(r'Critical Alpha ($\alpha$)', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)

# plot MSE to time awake
fig, ax = plt.subplots()
x = df['RT']
x = x.values.reshape(-1, 1)
y = df["'AF3_MSE'"]
# y = y.values.reshape(-1, 1)
model = BayesianRidge().fit(x, y)
sns.scatterplot(x='RT', y="'AF3_MSE'", data=df, hue='timeAwake')
plt.plot(x, model.predict(x), color='black')
plt.text(0.75, 0.15, 'Slope: {:.3f}'.format(model.coef_[0]), transform=ax.transAxes, fontsize=22)
plt.text(0.75, 0.1, 'Intercept: {:.3f}'.format(model.intercept_), transform=ax.transAxes, fontsize=22)
plt.text(0.75, 0.05, 'R-squared: {:.3f}'.format(model.score(x, y)), transform=ax.transAxes, fontsize=22)
plt.show()
ax = plt.gca()
# ax.set_title('Critical Alpha Metric over Reaction Time', fontsize=22)
leg = plt.legend()
leg.set_title('Time Awake (h)', prop={'size': 18})
ax.set_xlabel('Reaction Time (s)', fontsize=22)
ax.set_ylabel(r'AF3_MSE', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)


# Violin Plot of MSE over Time Awake:
fig, ax = plt.subplots()
sns.violinplot(x='timeAwake', y="'AF3_MSE'", data=df, ax=ax)
# grouped = df.groupby('timeAwake')
# regression_lines = {}
# p_values = {}
# for name, group in grouped:
#     X = group['timeAwake'].values.reshape(-1, 1)
#     y = group["'AF3_MSE'"]
#     model = sm.OLS(y, sm.add_constant(X)).fit()
#     regression_lines[name] = model.params[0] + model.params[1] * X
#     p_values[name] = model.pvalues[1]
#     sns.lineplot(x=X.flatten(), y=regression_lines[name].flatten(), ax=ax, color='black')
# for i, timeAwake in enumerate(p_values.keys()):
#     plt.annotate("p = {:.3f}".format(p_values[timeAwake]),(timeAwake, regression_lines[timeAwake][0]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=16)
plt.show()
ax.set_title('AF3 MSE over Time Awake', fontsize=22)
ax.set_xlabel('Time Awake (h)', fontsize=22)
ax.set_ylabel(r'AF3_MSE', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/