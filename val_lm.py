
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import  pandas as pd

df = pd.read_csv("D:\!Code\datathon\dataset\datathon.csv")
cy_cols = list(df.filter(regex ="cy|CY").columns)
elements = ["POP_EST_CY", "POP_DENSITY_CY","ADJUSTED_LIFECYCLE_PY","PCT_OF_LIFECYCLE_REMAINING","PCT_OF_LIFECYCLE_REMAINING", "LIFECYCLE_PRE_PEAK_POST"]
new_cy_cols = [i for i in cy_cols if i not in elements]

x_train = df.drop(new_cy_cols, axis=1)
print(x_train.columns)
print(x_train.shape)
y_train = df["qty_sold_cy_diy"]
y_train2 = df["qty_sold_cy_difm"]
print(y_train.columns)
lm = LinearRegression()
scores = cross_val_score(lm, x_train, y_train, scoring="neg_mean_squared_error", cv=5)
best_scores = max(scores)
print(best_scores)