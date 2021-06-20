#   Technical Test MP Data
##  Author : Wessim Slimi
##  Date : Jun 20  2021


## 1. Scripts
* train.py : trains a gradient boosting model and a shap explainer object and save them respectively into ./model and ./pickles folders
* app.py   : runs flask web application to input data manually and perform predictions
* conf.py  : contains all the necessary configurations to be used by train.py and app.py

## 2. Folders
* model : contains saved models 
* pickles : contains saved Shap values explainers
* images : contains various images,  containing model performance and prediction shap value
* templates : contains .html templates
* catboost_info : temp files containing necessary information about the trained model

## 3. Other files
* Test Technique.ipynb : Notebook containing analysis about training and testing the model
* nba_logreg.csv : csv file containig data


## How to run:
1. run pip install -r requirements.txt to download all the libraries
2. run python train.py to train the model
3. run python app.py to launch the Flask web server. Default url is http://localhost:3000/predict <br>
you can specify which model you want to use, you will just need to add --timestamp=<timestamp> example : 
python app.py --timestamp='2021_06_20-10:44:43_PM'