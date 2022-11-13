import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('D:\!Code\datathon\dataset\datathon.csv')
cy_cols = list(df.filter(regex='cy|CY').columns)
elements = ['POP_EST_CY', 'POP_DENSITY_CY', 'ADJUSTED_LIFECYCLE_PY', 'PCT_OF_LIFECYCLE_REMAINING',
            'PCT_OF_LIFECYCLE_REMAINING',
            'LIFECYCLE_PRE_PEAK_POST']
new_cy_cols = [i for i in cy_cols if i not in elements]
x_train = df.drop(new_cy_cols, axis=1)
y_train = df['qty_sold_cy_diy']
y_train2 = df['qty_sold_cy_difm']

x_train = x_train.dropna(axis=1, how='all').fillna(0)

from sklearn import preprocessing

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

x_train.head(1000).to_csv("sample.csv")

lm = LinearRegression()
from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(lm, x_train, y_train, scoring='neg_mean_squared_error', cv=5)
best_scores_diy = max(scores1)
print(best_scores_diy)

scores2 = cross_val_score(lm, x_train, y_train2, scoring='neg_mean_squared_error', cv=5)
best_scores_difm = max(scores2)
print(best_scores_difm)
