import pandas as pd

from numpy import mean

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from xgboost import XGBClassifier

file = pd.ExcelFile("path/to/file/onp.xlsx")
file.sheet_names
df1=file.parse('Tabelle1')
dataset = df1
array = dataset.values

X = array[:,0:7]
y = array[:,7]

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X = min_max_scaler.fit_transform(X)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=50)))
models.append(('MLP', MLPClassifier(solver='adam', hidden_layer_sizes=(64), max_iter=500, activation='relu', random_state=1)))
models.append(('DTC (CART)', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('GBC (XGBoost)', XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric="logloss", tree_method="exact", scale_pos_weight=13.4)))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy') # Alternative: scoring='roc_auc'
    results.append(cv_results)
    names.append(name)
    print(name + ' - mittlerer AUC-Score: %.3f' % mean(cv_results) + ' - Standardabweichung: %.3f' % cv_results.std())
