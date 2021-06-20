import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
import copy
import shap
import pickle
import os


# custom conf file containing variables definitions and paths
import configurations as conf
import utils


# these features are the highly correlated features. See notebook for details


def data_prep(data):
    res_data = copy.deepcopy(data)
    res_data['3P%'] = res_data['3P%'].fillna(0)
    res_data[conf.target] = res_data[conf.target].astype(int)
    res_data.drop(conf.cols_to_drop, axis=1, inplace=True)
    return res_data


def plot_shap(shap_values, X, shap_path):
    plt.figure()
    fig = shap.summary_plot(shap_values, X, show=False)
    plt.savefig(shap_path)
    plt.close(fig) 

def main():
    for dir in conf.dirs:
        try:
            os.makedirs(dir)
        except FileExistsError:
            # directory already exists
            pass

    # reading data
    print('- Reading Data')
    raw_data = pd.read_csv(conf.data_path)

    # dataprep
    print('- Data Prep')
    df = data_prep(raw_data)
    X = df[conf.features].drop(conf.target, axis=1)
    y = df[conf.target]

    # splitting training an testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=conf.test_split)

    # class weights for inbalanced data
    class_weights = utils.get_class_weights(y)
    print('- Traning model')

    # define model
    model = CatBoostClassifier(
        iterations = conf.iterations,
        loss_function = conf.loss,
        learning_rate=conf.learning_rate,
        depth = conf.depth,
        class_weights = class_weights,
        early_stopping_rounds = conf.early_stopping_rounds,
)
    # train model
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    # save model
    model.save_model(conf.model_path,
           format="cbm",
           export_parameters=None,
           pool=None)
    print('- Model was saved to {}'.format(conf.model_path))

    # making predictions on testing set
    y_pred = model.predict(X_test)

    # create perf image
    utils.plot_evaluation(model, y_test, y_pred, conf.image_path)
    print('- Model perfs saved to {}'.format(conf.image_path))

    # creating explainer to be used for shap values
    explainer = shap.Explainer(model, feature_names=X_test.columns.tolist())

    # plot Summary of the variables
    # summarize the effects of all the features
    shap_values = explainer(X_train)
    plot_shap(shap_values, X_train, conf.shap_fig_path)
    print('- Shap figure saved to {}'.format(conf.shap_fig_path))

    # saving Shap values explainer object to file
    with open(conf.explainer_path, 'wb') as f:
        pickle.dump(explainer, f)
    print('- Explainer saved to {}'.format(conf.explainer_path))

    # success message
    print('Model Trained Succesfully !')


if __name__ == '__main__':
    main()