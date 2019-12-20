#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report


#%%
data_orig = pd.read_csv('./data/crx.data', header=None)


#%%
data = data_orig.copy()
data.replace('?', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)
data.drop([13], axis=1, inplace=True)

data[1] = pd.to_numeric(data[1])

for col in data:
    if data[col].dtypes == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])


#%%
X, Y = data.values[:, :14], data.values[:, 14:]
Y = np.ravel(Y)


#%%
DTL = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)
# DTL.fit(X, Y)
DTL_Y_hat = cross_val_predict(DTL, X, Y, cv=10)
DTL_matrix = confusion_matrix(Y, DTL_Y_hat)
DTL_report = classification_report(Y, DTL_Y_hat)

print(DTL_matrix)
print(DTL_report)


#%%
MLP = MLPClassifier(hidden_layer_sizes=(100,100), activation='tanh', solver='adam', learning_rate='constant',
                    learning_rate_init=0.001, max_iter=1000, random_state=0)
# MLP.fit(X, Y)
MLP_Y_hat = cross_val_predict(MLP, X, Y, cv=10)
MLP_matrix = confusion_matrix(Y, MLP_Y_hat)
MLP_report = classification_report(Y, MLP_Y_hat)

print(MLP_matrix)
print(MLP_report)


#%%
RF = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
# RF.fit(X, Y)
RF_Y_hat = cross_val_predict(RF, X, Y, cv=10)
RF_matrix = confusion_matrix(Y, RF_Y_hat)
RF_report = classification_report(Y, RF_Y_hat)

print(RF_matrix)
print(RF_report)


#%%
logistic = LogisticRegression(solver='newton-cg', C=6000, tol=1)
RBM = BernoulliRBM(learning_rate=0.01, n_iter=10, n_components=100, random_state=0)

RBM_classifier = Pipeline(steps=[('RBM', RBM), ('Logistic', logistic)])
RBM_Y_hat = cross_val_predict(RBM_classifier, X, Y, cv=10)
RBM_matrix = confusion_matrix(Y, RBM_Y_hat)
RBM_report = classification_report(Y, RBM_Y_hat)

print(RBM_matrix)
print(RBM_report)


#%%
