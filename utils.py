from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import configurations as conf
from tqdm.notebook import tqdm
import copy
from scipy import stats
import shap

# Defining functions

def get_class_weights(arr, labels = [0,1]):
    """
    Calculates class weights for multi class 
    
    Parameter : 
        arr (numpy.array) : array containing target variable
        labels (list) : list of classes
    
    Returns:
        class_weights (dict) : dict containing class label as key 
                               and it's corresponding weight as values  
    """
    class_weights = {}
    for label in labels:
        try:
            class_weights[label] = len(arr) / (2*len(arr[arr==label]))
        except ValueError as e:
            print('Error Could not calculate class weights. There is maybe a division by 0 please verify that all the classes exists. Traceback below \n'.format(e))
    return class_weights

def plot_evaluation(model, y_test, y_pred, file_path):
    """
    Plot 3 plots : Raw Confusion Matrix, Normalized Confusion Matrix, and Roc Curve

    Parameters:
        - model (can be any that accepts a fit method classifier) : used model
        - y_test (numpy array) : testing array containing ground truth
        - y_pred (numpy array) : predicted classes
    Returns:
        - None
    """
    confusion_matrix(y_test, y_pred)
    conf_mat = confusion_matrix(np.array(y_test), np.array(y_pred))
    N_0 = len(y_test.values[y_test.values ==0])
    N_1 = len(y_test.values[y_test.values ==1])
    mat = np.zeros((2,2))
    mat[0] = conf_mat[0]/(conf_mat[0,0]+conf_mat[0,1])
    mat[1] = conf_mat[1]/(conf_mat[1,0]+conf_mat[1,1])
    # Confusion Matrix
    plt.figure(figsize=(18,4))
    plt.subplot(131)
    plt.title('True Confusion matrix \n F1:{:.2f}'.format(f1_score(np.array(y_test), np.array(y_pred))))
    sns.heatmap(conf_mat, annot=True)
    plt.subplot(132)
    plt.title('Weighted Confusion matrix \n F1:{:.2f}'.format(f1_score(np.array(y_test), np.array(y_pred))))
    sns.heatmap(mat, annot=True)
    # ROC curse and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_value = auc(fpr, tpr)
    plt.subplot(133)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='(area = {:.3f})'.format(auc_value))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve \n AUC : {}'.format(auc_value))
    plt.legend(loc='best')
    plt.savefig(file_path)
  

def box_cox_transform(data):
    """
    Performs a box cox transformation. This transformation is used on the features, it makes them more normal.

    Parameters:
        - data (dataframe) : containing data

    Returns:
        - df (dataframe) : containing transformed data
        - dict_lambdas (dict) : dict containing corresponding lambda to each variable. Used for reverse transformation
    """
    df = copy.deepcopy(data)
    dict_lambdas = {}
    for col in tqdm(data.columns):
        if col == conf.target:
            pass
        else:
            df[col], dict_lambdas[col] = stats.boxcox(data[col].values + np.abs(np.min(data[col].values))+1)
    return df, dict_lambdas

def plot_shap(model, X, explainer, player_name=''):
    """
    Create a Shap prediction plot
    
    Parameters:
        - model : Model
        - X : (Array) : Input data. used for prediction 
        - explainer (object) : explainer object used to calculate shap values
    Returns:
        - None
    """
    # calculating shap values
    shap_values = explainer(X)  
    # visualize the first prediction's explanation
    plt.figure()
    fig = shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig('{}-{}.png'.format(conf.waterfall_fig_path, player_name), format='png')