# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
import csv
import dvc.api
import yaml

# %%
#from google.colab import drive
#drive.mount('/content/drive/')


# %%
#cd /content/drive/My Drive/Colab Notebooks/


# %%
train_df_path = "Data/creditcard.csv"
data_fraud = pd.read_csv(train_df_path)
params = yaml.safe_load(open("params.yaml"))["params"]
C=params["C"]
kernel=params["kernel"]
degree=params["degree"]
gamma=params["gamma"]
'''
with dvc.api.open(
        'creditcard.csv',
        remote='remote'
        ) as fd:
    data_fraud = pd.read_csv(fd)
'''
# %%
#print(data_fraud)


# %%
y=data_fraud['Class']
x=data_fraud.drop('Class',axis=1)


# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#x_train.head()
x_test.to_csv("Data/x_test.csv")
y_test.to_csv("Data/y_test.csv")

# %%
SVM = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
SVM.fit(x_train,y_train)
Pkl_Filename = "Models/Fraud_SVM.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(SVM, file)


# %%
y_train_pred = SVM.predict(x_train)
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
'''
predictions_SVM = SVM.predict(x_test)
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, predictions_SVM)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)
'''


# %%
plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
#plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
#plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.title("AUC(ROC curve)")
#plt.grid(color='black', linestyle='-', linewidth=0.5)

#plot_confusion_matrix(SVM, x_test, y_test, normalize='true')
#plt.show()


