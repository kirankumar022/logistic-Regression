import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
election = pd.read_csv("E:/Assignments/Assignment week 11/logistic/assignments/election_data.csv")

#removing CASENUM
e1 = election.drop('Election-id', axis = 1)
e1.head(11)
e1.describe()
e1.isna().sum()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(e1.drop('Result',axis=1),e1['Result'], test_size=0.25, random_state=101)
from sklearn.linear_model import LogisticRegression
logit_model= LogisticRegression()
logit_model.fit(X_train,y_train)

predictions = logit_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
logit_model

pred = logit_model.predict(e1.iloc[ :, :-1 ])

fpr, tpr, thresholds = roc_curve(e1['Result'], pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


