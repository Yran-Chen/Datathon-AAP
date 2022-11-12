import pandas as pd
import datetime
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

from dataset_for_classifier import  DatasetPool

parser = argparse.ArgumentParser(description='construct pretrain settings.')
# parser.add_argument('--op', type=str, default = None)
# parser.add_argument('--selected', type=str, default = None)
parser.add_argument('--percent', type=float, default = None)
parser.add_argument('--name',type=str,default=None)
parser.add_argument('--rm_cache',type=bool,default=None)
args = parser.parse_args()

s = "SKU_NUMBER,STORE_NUMBER,PART_TYPE,PLATFORM_CLUSTER_NAME,APPLICATION_COUNT, POP_EST_CY, POP_DENSITY_CY, PCT_WHITE, AGE,PCT_COLLEGE,PCT_BLUE_COLLAR,MEDIAN_HOUSEHOLD_INCOME,ESTABLISHMENTS, SKU_EXISTENCE_PY, SKU_STORE_PDQ_PY,TOTAL_VIO_PY,LOST_QTY_PY,AVG_CLUSTER_UNIT_SALES_PY,AVG_CLUSTER_LOST_SALES_PY, ROAD_QUALITY_INDEX,VIO_COMPARED_TO_CLUSTER_PY,ADJUSTED_AVG_CLUSTER_SALES_PY,QTY_SOLD_PPY,QTY_SOLD_PY,AVG_CLUSTER_TOTAL_SALES_PY,SALES_SIGNAL_PY,LIFECYCLE_PY,ADJUSTED_LIFECYCLE_PY,PCT_OF_LIFECYCLE_REMAINING,LIFECYCLE_PRE_PEAK_POST,ADJ_AVG_CLUSTER_LOST_SALES_PY,ADJ_AVG_CLUSTER_TOTAL_SALES_PY,ppy_unit_sales,py_unit_sales,PROJECTED_GROWTH_PCT_PY,OTHER_UNIT_PLS_LOST_SALES_PPY,OTHER_UNIT_PLS_LOST_SALES_PY,MPOG_ID,qty_sold_py_difm,qty_sold_py_diy,py_gross_sales,ppy_gross_sales,py_sales_cost,ppy_sales_cost,py_qty_sold_transfer,ppy_qty_sold_transfer,py_qty_sold_on_hand,ppy_qty_sold_on_hand"

s = s.split(",")
p = list()
for i in s:
    p.append(i.strip(" "))
print(p)

PARAM_MAIN = {
    'name':"main",
    'data_dir': r'D:\!Code\datathon\dataset',
    'save_dir': r"D:\!Code\datathon\Savefile",
}

PARAM_TEST = {
    'name':"testing",
    'data_dir': r'D:\!Code\datathon\dataset_test',
    'save_dir': r"D:\!Code\datathon\Savefile",
    'rm_cache': False,
}
if __name__ == "__main__":

    run_cmd = {
        "encoder_label": ["SKU_NUMBER", "MPOG_ID", "PART_TYPE"],
        "if_overwrite": False,
        "pca":40,
        "split":0.3,
        "train_dataset_name":"data_split_2y_sel1",
        "y":["qty_sold_cy_difm","qty_sold_cy_diy",
             "qty_sold_py_diy",	"qty_sold_py_difm",
             "qty_sold_ppy_diy"	,"qty_sold_ppy_difm"],
        "x":p,
        "training_pairs": [
            ["qty_sold_ppy_difm","qty_sold_py_difm","qty_sold_cy_difm"],
            ["qty_sold_ppy_diy","qty_sold_py_diy","qty_sold_cy_diy"],

        ]
    }
    run_cmd_main = {
        "encoder_label": ["SKU_NUMBER", "MPOG_ID", "PART_TYPE"],
        "if_overwrite": False,
        "pca":40,
        "split":0.3,
        "train_dataset_name":"data_split_2y_sel1_whole",
        "y":["qty_sold_cy_difm","qty_sold_cy_diy",
             "qty_sold_py_diy",	"qty_sold_py_difm",
             "qty_sold_ppy_diy"	,"qty_sold_ppy_difm"],
        "x":p,
        "training_pairs": [
            ["qty_sold_ppy_difm","qty_sold_py_difm","qty_sold_cy_difm"],
            ["qty_sold_ppy_diy","qty_sold_py_diy","qty_sold_cy_diy"],

        ]
    }
    if args.percent is not None:
           PARAM_TEST['percent'] = args.percent
    if args.name is not None:
           PARAM_TEST['name'] = args.name
    if args.rm_cache is not None:
           PARAM_TEST['rm_cache'] = args.rm_cache

    # tst_default = DatasetPool(PARAM_DEFAULT)
    # tst_default.data_forward(**run_cmd)

    tst_be = DatasetPool(PARAM_TEST)
    tst_be.run(**run_cmd)
    tst_be.loading_training_(**run_cmd)

    # tst_main = DatasetPool(PARAM_MAIN)
    # tst_main.run(**run_cmd_main)
