# Plot AUC curves for classifiers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
  3 Features
'''
# feature_whole = pd.read_csv('Data/precision_recall_3Features_filtered_False.csv')
# feature_subset = pd.read_csv('Data/precision_recall_3Features_filtered_True.csv')
# n_feature = 3
# whole_auc = {'mlp':0.935, 'logistic_regression':0.930, 'svc':0.905} # scores need to be added manually
# subset_auc = {'mlp':0.891, 'logistic_regression':0.872, 'svc':0.887} # scores need to be added manually

'''
  7 Features
'''
feature_whole = pd.read_csv('Data/precision_recall_7Features_filtered_False.csv')
feature_subset = pd.read_csv('Data/precision_recall_7Features_filtered_True.csv')
n_feature = 7
whole_auc = {'mlp':0.921, 'logistic_regression':0.923, 'svc':0.912} # scores need to be added manually
subset_auc = {'mlp':0.882, 'logistic_regression':0.887, 'svc':0.885} # scores need to be added manually

set_indx = 1 if n_feature == 7 else 2

print(feature_whole, feature_subset)
print(feature_whole.columns, feature_subset.columns)

for idx_feature_whole in feature_whole.index:
    feature_whole.loc[idx_feature_whole,'precisions'] = \
        np.array(feature_whole.loc[idx_feature_whole,'precisions'].replace('[','').replace(']','').split()).astype(np.float)

    feature_whole.loc[idx_feature_whole,'recalls'] = \
        np.array(feature_whole.loc[idx_feature_whole,'recalls'].replace('[','').replace(']','').split()).astype(np.float)

for idx_feature_subset in feature_subset.index:
    feature_subset.loc[idx_feature_subset,'precisions'] = \
        np.array(feature_subset.loc[idx_feature_subset,'precisions'].replace('[','').replace(']','').split()).astype(np.float)

    feature_subset.loc[idx_feature_subset,'recalls'] = \
        np.array(feature_subset.loc[idx_feature_subset,'recalls'].replace('[','').replace(']','').split()).astype(np.float)


feature_whole_lr_recalls = feature_whole.iloc[0,2]
feature_whole_lr_precisons = feature_whole.iloc[0,1]
feature_whole_svc_recalls = feature_whole.iloc[1,2]
feature_whole_svc_precisons = feature_whole.iloc[1,1]
feature_whole_nn_recalls = feature_whole.iloc[2,2]
feature_whole_nn_precisons = feature_whole.iloc[2,1]

feature_subset_lr_recalls = feature_subset.iloc[0,2]
feature_subset_lr_precisons = feature_subset.iloc[0,1]
feature_subset_svc_recalls = feature_subset.iloc[1,2]
feature_subset_svc_precisons = feature_subset.iloc[1,1]
feature_subset_nn_recalls = feature_subset.iloc[2,2]
feature_subset_nn_precisons = feature_subset.iloc[2,1]

fig = plt.figure()
dataset_str = '(Dataset ' + str(set_indx) + ')'
plt.plot(feature_whole_nn_recalls,feature_whole_nn_precisons, c='black', label=('Neural Network ' + dataset_str + ', AUC=' + str(whole_auc['mlp'])))
plt.plot(feature_whole_lr_recalls,feature_whole_lr_precisons, c='orange', label=('Logistic Regression ' + dataset_str + ', AUC=' + str(whole_auc['logistic_regression'])))
plt.plot(feature_whole_svc_recalls,feature_whole_svc_precisons, c='green', label=('Support Vector Machine ' + dataset_str + ', AUC=' + str(whole_auc['svc'])))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
fig.savefig('../Figures/' + str(n_feature) + '_features_precision_recall_whole.png')
plt.show()

fig = plt.figure()
subset_str = '(Subset ' + str(set_indx) + ')'
plt.plot(feature_subset_nn_recalls,feature_subset_nn_precisons, c='black', label=('Neural Network ' + subset_str + ', AUC=' + str(subset_auc['mlp'])))
plt.plot(feature_subset_lr_recalls,feature_subset_lr_precisons, c='orange', label=('Logistic Regression ' + subset_str + ', AUC=' + str(subset_auc['logistic_regression'])))
plt.plot(feature_subset_svc_recalls,feature_subset_svc_precisons, c='green', label=('Support Vector Machine ' + subset_str + ', AUC=' + str(subset_auc['svc'])))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
fig.savefig('../Figures/' + str(n_feature) + '_features_precision_recall_subset.png')
plt.show()

