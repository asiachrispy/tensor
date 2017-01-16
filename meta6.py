__author__ = 'huangzhong'
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

dataset =  loadtxt('',delimiter=",")
X = data[:,0:8]
Y = data[:,8]

model = XGBClassifier()
kfold = KFold(n=len(X),n_folds=10,random_state=7)
results = cross_val_score(model,X,Y,cv=kfold)
print ("acc:%.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))