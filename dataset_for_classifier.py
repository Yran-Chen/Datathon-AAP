import numpy as np
import pandas as pd
import os
import json
import codecs
import pickle
from time import *
import collections
from sklearn import preprocessing
import sklearn
import copy
from multiprocessing import Process, Pool

model_param = {
    'model':'GradientBoostingClassifier',

    "model_settings": {
        "loss": "deviance",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "subsample": 1.0,
        "max_features": 0.8,
        "validation_fraction": 0.1,
        "random_state": 42
    }
}

OP_DICT = [
    # 'add','sub','mul','div',
    # 'sqrt','log','square','zscore','sigmoid',
    'square',
]

PARAM_TEST = {
    'name':'test',
    'data_dir': r'D:\!DTStack\Dataset\UCI_\ml\machine-learning-databases',
    'save_dir': r"D:\!DTStack\Savefile",
    'threshold':0.01,
    'operator':OP_DICT,
    'selected': '!f',
    'pre_model_param':model_param,
    'percent':1.0,
}

from common.utils import LogHandler,log,_read_pickle,_save_pickle,create_dir
logHandler = LogHandler()._log

PROCESS_NUM = 8

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cb


"""
FOR DEBUG ONLY.
"""
np.set_printoptions(threshold=np.inf)
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class DatasetPool():

    def __init__(self,param):
        self.param = param
        self.name = param['name']
        self.data_dir = param['data_dir']
        self.save_dir = param['save_dir']

        self.percent = param['percent'] if "percent" in param.keys() else None
        self.dataset = None
        self.dataset_dir = {}
        self.pool = None
        self.dataset_input = {}
        self.learner_if_clean = False
        self.feature_pool = collections.defaultdict(list)
        self.training_dataset = {}


    @log(_log = logHandler)
    def data_forward(self,*args,**kwargs):
        self.dataset_preprocessing(**kwargs)
        # self.operator_pretraining(**kwargs)
        self.clean_data(**kwargs)

    def feature_eng(self,*args,**kwargs):
        # self.dataset_preprocessing(**kwargs)
        self.loading_training_(**kwargs)

    @log(_log = logHandler)
    def data_split(self,*args,**kwargs):
        print("starting...")
        self.dataset_preprocessing(**kwargs)
        self.split_data_r(**kwargs)
        # self.run_LFE_learner(**kwargs)
        return

    @log(_log = logHandler)
    def run(self,*args,**kwargs):
        print("starting...")
        self.loading_training_(**kwargs)
        # self.run_LFE_learner(**kwargs)
        return

    def preload_data(self,**kwargs):
        self.dataset_preprocessing(**kwargs)
        self.clean_data(**kwargs)
        self.split_data_r(**kwargs)

    @log(_log = logHandler)
    def loading_training_(self,**kwargs):
        name = kwargs["train_dataset_name"] if "train_dataset_name" in kwargs.keys() else None
        y_ = ["qty_sold_cy_difm","qty_sold_cy_diy"]
        path_list = []
        if os.path.exists(os.path.join(self.save_dir, name)):
            for yi in y_:
                for csv_x,csv_Y in [["X_train.csv","y_train.csv"]]:
                    p1 = os.path.join(self.save_dir, name, yi, csv_x)
                    p2 = os.path.join(self.save_dir, name, yi, csv_Y)
                    self.training_dataset[yi] = [pd.read_csv(p1),pd.read_csv(p2,header = None)]
        else:
            self.preload_data(**kwargs)
        # print(self.training_dataset)


    def split_data_r(self,**kwargs):
        name = kwargs["train_dataset_name"] if "train_dataset_name" in kwargs.keys() else None
        for ni in self.dataset:
            operator_dataset = self.dataset_input[ni]
            begin_time = time()
            if "percent" in kwargs.keys():
                operator_dataset = operator_dataset.sample(frac=kwargs["percent"])

            if "split" in kwargs.keys():
                for x_1,x_2,y_ in kwargs["training_pairs"]:
                    xidx = copy.deepcopy(self.feature_pool["x"])
                    xidx.append(x_1)
                    xidx.append(x_2)
                    yidx = y_

                    xdf = operator_dataset[xidx]
                    ydf = operator_dataset[yidx]
                # print(x,y)
                    X_train, X_test, y_train, y_test = \
                    sklearn.model_selection.train_test_split(xdf, ydf,
                                                                test_size=kwargs['split']
                                                                )
                    # print(X_test,y_test)
                    xt_path = os.path.join(self.save_dir, name,y_ ,'X_train.csv')
                    create_dir(xt_path)
                    self.save_csv_from_df(xt_path, X_train)
                    xts_path = os.path.join(self.save_dir, name, y_, 'X_test.csv')
                    create_dir(xts_path)
                    self.save_csv_from_df(xts_path, X_test)
                    yt_path = os.path.join(self.save_dir, name,y_ ,'y_train.csv')
                    create_dir(yt_path)
                    self.save_csv_from_df(yt_path, y_train)
                    yts_path = os.path.join(self.save_dir, name, y_, 'y_test.csv')
                    create_dir(yts_path)
                    self.save_csv_from_df(yts_path, y_test)

                    self.training_dataset[y_] = [X_train,y_train, X_test, y_test]


            end_time = time()
            print('Time Usage for op {} dataset create is : {:.2f}'.format(name, (end_time - begin_time)))
            logHandler.info('Time Usage for op {} dataset create is : {:.2f}'.format(name, (end_time - begin_time)))

    def prepare_operator_dataset(self,oprtr:str)->pd.DataFrame:

        print ('@{} Dataset Processing...'.format(oprtr))
        logHandler.info('@{} Dataset Processing...'.format(oprtr))

        if self.learner_if_clean:
            df_lfe_table = self.clean_data(df_lfe_table,range = self.cleaned_range)

        # threshed_oprtr_performance = self.threshold_forward(df_lfe_table)
        # return self.QSA_forward(oprtr,threshed_oprtr_performance)

        return
        # return self.split_data_sklearn()

    def clean_data(self,*args,**kwargs):
        for name in self.dataset:
            df_lfe_table = self.dataset_input[name]
            if "encoder_label" in kwargs.keys():
                for lb in kwargs["encoder_label"]:
                    labelendr = preprocessing.LabelEncoder()
                    df_lfe_table[lb] = labelendr.fit_transform(df_lfe_table[lb].values)
                # print(df_lfe_table[lb])
            c1 = set(df_lfe_table.columns)
            df_lfe_table=df_lfe_table.dropna(axis=1,how='all')
            c2 = set(df_lfe_table.columns)
            # print(c1-c2)
            self.dataset_input[name] = df_lfe_table
            if "if_overwrite" in kwargs.keys() and kwargs["if_overwrite"]:
                self.save_csv_from_df(self.dataset_dir[name],df_lfe_table)
        # return data[(data['performance']<range[0]) | (data['performance']>range[1])]

    def dataset_preprocessing(self,**kwargs):

        begin_time = time()

        self.dataset = self.load_dataset_name()
        logHandler.info(self.dataset)
        self.dataset_dir = self.load_dataset_dir()
        self.load_dataset_forward()
        self.feature_pool_create(**kwargs)

        end_time = time()
        print('Time Usage for data preprocessing is : {:.2f}'.format((end_time - begin_time)))
        logHandler.info('Time Usage for data preprocessing is : {:.2f}'.format((end_time - begin_time)))

    def feature_pool_create(self,**kwargs):
        self.feature_pool["y"] = kwargs["y"]
        self.feature_pool["x"] = kwargs["x"]
        # print (self.feature_pool)

    def load_dataset_forward(self):
        for name in self.dataset:
            self.dataset_input[name] = self.load_from_csv(self.dataset_dir[name])

    def feature_reduction(self,df_data:pd.DataFrame,n_components=49)->pd.DataFrame:
        from sklearn.decomposition import PCA

        if df_data.shape[1] <= n_components:
            return df_data

        else:
            pca = PCA(n_components=n_components)

        print(pca.explained_variance_ratio_)
        return df_data.fit_transform(df_data)

    def update_dataset_config(self)-> None:
        for name in self.dataset:
                begin_time = time()
                self.dataset_config[name] = self.gather_dataset_information(name)
                end_time = time()

    # dataset_name, dataset_shape, label
    def gather_dataset_information(self,dataset_name) -> dict:
        dataset_info = {}
        dataset_info['name'] = dataset_name
        _, label , shape = self.load_dataset_data(dataset_name)
        dataset_info['label'] = label
        dataset_info['shape'] = shape
        return dataset_info

    def run_training_model(self,df_raw_data,dataset_name,label):

        # df_raw_data = df_raw_data.sample(frac=1)
        df_raw_data_label = copy.deepcopy( (df_raw_data.iloc[:,-1].apply(str) ) )

        #trans to 1vR task.
        df_raw_data_label[df_raw_data_label!=label] = 'non_label'
        labelendr = preprocessing.LabelEncoder()

        data_y = labelendr.fit_transform(df_raw_data_label)
        data_x = df_raw_data.iloc[:,0:-1].values

        logHandler.info(str(data_x.shape))
        logHandler.info(data_x.mean())
        # logHandler.info(str(data_y))

        # # fit into [example , feature]
        # labels = np.array(labels).reshape(len(labels), 1)
        # onehot = preprocessing.OneHotEncoder()
        # onehot_label = onehot.fit_transform(labels)
        # np_data_y = onehot_label.toarray()

        # feed into model.
        model = eval(self.pre_model)
        clf_svc_cv = model(**self.pre_model_setting)
        # clf_svc_cv = GradientBoostingClassifier()

        # cbmodel = cb.CatBoostClassifier(iterations=100,silent=True)
        # cbmodel.fit(data_x,data_y,plot = False,silent = True)
        # scores_clf_cv = cbmodel.score(data_x, data_y)

        """
        for fastbacktest.
        """
        # clf_svc_cv.fit( X=data_x, y=data_y )
        # scores_clf_cv = clf_svc_cv.score( X=data_x, y=data_y)

        # print(  np.sort(clf_svc_cv.feature_importances_)   )
        # print (  np.argsort(clf_svc_cv.feature_importances_)  )
        """
        for cv backtest.
        """

        scores_clf_cv = cross_val_score(clf_svc_cv, data_x, data_y, cv = 5)
        #
        print(scores_clf_cv)
        print("Accuracy: %f (+/- %0.4f)" % (scores_clf_cv.mean(), scores_clf_cv.std() * 2))

        return scores_clf_cv.mean()

    def load_dataset_name(self)->list:
        retdir = []
        tmpdir = os.listdir(self.data_dir)
        selected = self.param['selected'] if 'selected' in self.param.keys() else None
        if selected is None:
            return tmpdir
        else:
            for i in tmpdir:
                if i.startswith(selected):
                    retdir.append(i)
            return retdir

    def load_dataset_dir(self):
        dict_datasetdir = {}
        for name in self.dataset:
                    __path = os.path.join(self.data_dir, name)
                    print(__path)
                    if __path.endswith('csv'):
                        dict_datasetdir[name]=__path
        return dict_datasetdir

    def load_dataset_config(self) -> dict:
        load_path = os.path.join(self.save_dir,self.name,'dataset_config.pickle')
        if os.path.exists(load_path):
            print('Succ load dataset conf.')
            return _read_pickle(load_path)
        else:
            return {}

    def save_dataset_config(self):
        save_path = os.path.join(self.save_dir,self.name,'dataset_config.pickle')
        _save_pickle(self.dataset_config,save_path)


    #fetch all data through dataset name.
    def load_dataset_data(self,dataset_name:str)-> (pd.DataFrame,list,list):
        if dataset_name in self.dataset_input.keys():
            return self.dataset_input[dataset_name],None,None
        df_data = pd.DataFrame()
        for __dataset_dir in self.dataset_dir[dataset_name]:
            df_data = pd.concat([df_data,self.load_from_csv(__dataset_dir)],axis = 0)

        # For large memory.
        self.dataset_input[dataset_name] = df_data
        # print (df_data)
        # label = [ str(lbl) for lbl in df_data.iloc[:,-1].unique()]
        label = None
        shape = list(df_data.shape)
        return df_data,label,shape

    def save_csv_from_df(self,__path, df: pd.DataFrame) -> None:
        return df.to_csv(__path,  index = None)

    def load_from_csv(self,__path: str) -> pd.DataFrame:
        df_csv = pd.read_csv(__path)
        return df_csv

    """
    for hdfs pipeline
    """

if __name__ == '__main__':


    pa = DatasetPool(PARAM_TEST)
    # pa.dataset_preprocessing()
    # print(pa.dict_LFEtable)
    pa.run()
    # print (pa.dict_LFEtable)
    # print (pa.dataset_config)
