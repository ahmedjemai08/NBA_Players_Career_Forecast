from flask import Flask, render_template, request, redirect
from catboost import CatBoostClassifier
import numpy as np

import pickle
from matplotlib import pyplot as plt
import configurations as conf
import utils 
import os
import argparse

# get timestamp to choose wich model to use in production
parser = argparse.ArgumentParser(description='Please input timestamp')
parser.add_argument('--timestamp',help='timestamp', default='2021_06_21-12:05:59_AM')
args = parser.parse_args()
timestamp = str(args.timestamp)

# running flask server
app = Flask(__name__)

# defining the model
model = CatBoostClassifier()

# defining paths
model_path = './model/Catboost_Classifier-{}'.format(timestamp)
explainer_path = './pickles/explainer-{}'.format(timestamp)

# reading model
model.load_model(model_path)
# reading shap explainer
with open(explainer_path, 'rb') as pickle_file:
            explainer = pickle.load(pickle_file)


# configurations
app.config['UPLOAD_FOLDER'] = conf.picfolder


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        req = request.form
        input_data = list()
        missing = list()
        is_not_numeric = list()
        for k, v in req.items():
            if k in conf.features:
                input_data.append(v)
            if v=="":
                missing.append(k)
            if v.isnumeric() == False:
                if k == 'Name':
                    pass
                else:
                    is_not_numeric.append(k)
        if missing:
            feedback = "Missing fields for {}".format(', '.join(missing))
            print(feedback)
            return render_template("predict.html", feedback=feedback)

        if is_not_numeric:
            feedback = "Fields {} should be numeric type".format(', '.join(is_not_numeric))
            print(feedback)
            return render_template("predict.html", feedback=feedback)

        # making prediction
        player_name = req['Name']
        X_test = np.array(input_data).reshape(1,-1)
        prediction = model.predict(X_test)
        proba_predict = model.predict_proba(X_test)

        # making shap plot
        utils.plot_shap(model, X_test, explainer, player_name)
        shap_prediction = os.path.join(app.config['UPLOAD_FOLDER'], 'waterfall-{}.png'.format(player_name))

        return render_template(
            "display_prediction.html", prediction=prediction, 
            proba_of_failure='{:.2f}'.format(proba_predict[0][0]),
            proba_of_success='{:.2f}'.format(proba_predict[0][1]),
            player_name = req['Name'],
            shap_prediction = shap_prediction,
            
            )
        
    return render_template("predict.html")


if __name__ == '__main__':
    try:
        app.run(port=conf.port_number, debug=True)
    except OSError as e:
        print('Please check that port number : {} is not in use before running the server. Below Traceback error\n {}'.format(conf.port_number, e))
    finally:
        print('System Error: Could not launch the server')
