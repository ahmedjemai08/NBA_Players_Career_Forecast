from datetime import datetime
import os

# ============ Configuration file ============== #

# 1. Constants and variables
figsize = (20,20)
target = 'TARGET_5Yrs'
test_split = 0.2
cols_to_drop = [target]
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
feature_names =['GP','MIN','PTS','FGM','FGA','FG%','3P Made','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV']
cols_to_drop = ['Name', 'FGA', 'FGM']
features = ['GP', 'MIN', 'PTS', 'FG%', 'FTM', 
            'FTA', 'FT%', 'OREB','DREB', 'REB', 'STL', 
            'BLK', 'TOV', 'TARGET_5Yrs']
# 2. Paths
dirs = ['./static/images', './pickles', './model']
data_path = './nba_logreg.csv'
model_file_name = 'Catboost_Classifier_{}'.format(date)
perf_file_name = 'perfs-{}.png'.format(date)
explainer_file_name = 'explainer-{}.pickle'.format(date)
shap_fig_file_name = 'shap-{}.png'.format(date)
waterfall_fig_file_name = 'waterfall'
model_path =  os.path.join('./model',model_file_name)
image_path = os.path.join('./static/images', perf_file_name)
explainer_path = os.path.join('./pickles', explainer_file_name)
shap_fig_path = os.path.join('./static/images', shap_fig_file_name)
waterfall_fig_path = os.path.join('./static/images', waterfall_fig_file_name)
picfolder = os.path.join('./static', 'images')

# 3. Models params
learning_rate = 0.005
loss = 'Logloss'
iterations = 400
early_stopping_rounds = 50
depth = 4 


# 4. In production vars
model_path = './model/Catboost_Classifier-{}'.format(date)
explainer_path = './pickles/explainer-{}'.format(date)
port_number = 3000


if __name__ == '__main__':
    pass