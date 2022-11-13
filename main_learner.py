import pandas as pd
import numpy as np
import seaborn as sns
# Machine Learning Libraries
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from copy import deepcopy
import json
import statsmodels.api as sm
import os
import errno
import codecs
import time
import argparse
from common.utils import LogHandler,log,_read_pickle,_save_pickle,create_dir
logHandler = LogHandler()._log
from dataset_for_classifier import DatasetPool

def main_func():
    parser = argparse.ArgumentParser(description='construct pretrain settings.')
    # parser.add_argument('--op', type=str, default = None)
    # parser.add_argument('--selected', type=str, default = None)
    parser.add_argument('--percent', type=float, default = None)
    parser.add_argument('--name',type=str,default=None)
    parser.add_argument('--rm_cache',type=bool,default=None)
    args = parser.parse_args()

    s = "SKU_NUMBER,STORE_NUMBER,PART_TYPE,PLATFORM_CLUSTER_NAME,APPLICATION_COUNT, POP_EST_CY, POP_DENSITY_CY, PCT_WHITE, AGE,PCT_COLLEGE,PCT_BLUE_COLLAR,MEDIAN_HOUSEHOLD_INCOME,ESTABLISHMENTS, SKU_EXISTENCE_PY, SKU_STORE_PDQ_PY,TOTAL_VIO_PY,LOST_QTY_PY,AVG_CLUSTER_UNIT_SALES_PY,AVG_CLUSTER_LOST_SALES_PY, ROAD_QUALITY_INDEX,VIO_COMPARED_TO_CLUSTER_PY,ADJUSTED_AVG_CLUSTER_SALES_PY,QTY_SOLD_PPY,QTY_SOLD_PY,AVG_CLUSTER_TOTAL_SALES_PY,SALES_SIGNAL_PY,LIFECYCLE_PY,ADJUSTED_LIFECYCLE_PY,PCT_OF_LIFECYCLE_REMAINING,LIFECYCLE_PRE_PEAK_POST,ADJ_AVG_CLUSTER_LOST_SALES_PY,ADJ_AVG_CLUSTER_TOTAL_SALES_PY,ppy_unit_sales,py_unit_sales,PROJECTED_GROWTH_PCT_PY,OTHER_UNIT_PLS_LOST_SALES_PPY,OTHER_UNIT_PLS_LOST_SALES_PY,MPOG_ID,py_gross_sales,ppy_gross_sales,py_sales_cost,ppy_sales_cost,py_qty_sold_transfer,ppy_qty_sold_transfer,py_qty_sold_on_hand,ppy_qty_sold_on_hand"

    s = s.split(",")
    p = list()
    for i in s:
        p.append(i.strip(" "))

    PARAM_MAIN = {
        'name':"main",
        'data_dir': r'D:\!Code\datathon\dataset',
        'save_dir': r"D:\!Code\datathon\Savefile",
    }

    PARAM_TEST = {
        'name':"testing",
        'data_dir': r'D:\!Code\datathon\dataset',
        'save_dir': r"D:\!Code\datathon\Savefile",
        'rm_cache': False,
    }
    # Evaluation Libraries
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
    from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

    import warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

    # Load the data
    seed = 100

    run_cmd_main = {
        "encoder_label": ["SKU_NUMBER", "MPOG_ID", "PART_TYPE"],
        "if_overwrite": False,
        "pca": True,
        "split": 0.01,
        "strategy": "most_frequent",
        "if_pca":True,
        'ncp':32,
        "train_dataset_name": "data_main_new_v4_f",
        "y": ["qty_sold_cy_difm", "qty_sold_cy_diy",
              "qty_sold_py_diy", "qty_sold_py_difm",
              "qty_sold_ppy_diy", "qty_sold_ppy_difm"],
        "x": p,
        "training_pairs": [
            ["qty_sold_ppy_difm", "qty_sold_py_difm", "qty_sold_cy_difm"],
            ["qty_sold_ppy_diy", "qty_sold_py_diy", "qty_sold_cy_diy"],
        ]
    }

    tst_be = DatasetPool(PARAM_MAIN)
    tst_be.run(**run_cmd_main)

    run_cmd = {
        "encoder_label": ["SKU_NUMBER", "MPOG_ID", "PART_TYPE"],
        "if_overwrite": False,
        "pca": 40,
        "split": 0.1,
        "percent":0.01,
        "if_pca":True,
        'ncp':20,
        "train_dataset_name": "data_new_pca_dropna_s3",
        "y": ["qty_sold_cy_difm", "qty_sold_cy_diy",
              "qty_sold_py_diy", "qty_sold_py_difm",
              "qty_sold_ppy_diy", "qty_sold_ppy_difm"],
        "x": p,
        "training_pairs": [
            ["qty_sold_ppy_difm", "qty_sold_py_difm", "qty_sold_cy_difm"],
            ["qty_sold_ppy_diy", "qty_sold_py_diy", "qty_sold_cy_diy"],

        ]
    }
    # tst_be = DatasetPool(PARAM_TEST)
    # tst_be.run(**run_cmd)

    for yi in tst_be.training_dataset.keys():
        x_train = tst_be.training_dataset[yi][0]
        y_train = tst_be.training_dataset[yi][1]
        x_test =  tst_be.training_dataset[yi][2]
        y_test =  tst_be.training_dataset[yi][3]
        print(x_train.shape,y_train.shape)

        # lm = LinearRegression()
        # scores = cross_val_score(lm, x_train, y_train, scoring="neg_mean_squared_error", cv = 5)
        # best_scores = max(scores)
        # print("lm: ", best_scores)
        #
        # cp = {'objective':'reg:squarederror',"tree_method":"gpu_hist", "gpu_id":0}
        # xgbr = XGBRegressor(**cp)
        # xgbr.fit(x_train, y_train)
        # ypred = xgbr.predict(x_test)
        # mse = mean_squared_error(y_test, ypred)
        # print("MSE: % .2f" % (mse))

        from sklearn.model_selection import KFold
        from sklearn.model_selection import GridSearchCV
        from sklearn.feature_selection import RFE

        folds = KFold(n_splits=5, shuffle=True, random_state=100)
        # step-2: specify range of hyperparameters to tune
        hyper_params = [{"n_features_to_select": list(range(28, 32))}]
        # step-3: perform grid search
        # 3.1 specify model
        lm = LinearRegression()
        lm.fit(x_train, y_train)
        rfe = RFE(lm)
        # 3.2 call GridSearchCV()
        model_cv = GridSearchCV(estimator=rfe,
                                param_grid=hyper_params,
                                scoring= "neg_mean_squared_error",
        cv = folds,
        verbose = 1,
        return_train_score = True)
        # fit the model
        model_cv.fit(x_train, y_train)
        print(model_cv.best_score_)


        xptest = (x_test.iloc[:,-1] +  x_test.iloc[:,-2]) / 2
        mse_base = mean_squared_error(y_test, xptest)
        print('MSE base: % .2f' % (mse_base))


if __name__ == '__main__':
    main_func()