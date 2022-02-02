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
from sklearn.metrics import confusion_matrix
import json

x_test = pd.read_csv('Data/x_test.csv')
x_test = x_test.iloc[1: , :]
y_test = pd.read_csv('Data/y_test.csv')
y_test = y_test.iloc[1: , :]
x_test = x_test.iloc[: , 1:]
y_test=y_test.iloc[: , 1:]
Pkl_Filename = "Models/Fraud_SVM.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    SVM = pickle.load(file)

predictions_SVM = SVM.predict(x_test)
print(predictions_SVM)

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, predictions_SVM)
print(test_fpr)
print(test_tpr)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic',fontsize=22)
a=plt.plot(test_fpr, test_tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate',fontsize=22)
plt.xlabel('False Positive Rate',fontsize=22)
#plt.rcParams["font.size"] = "50"
plt.show()
plt.savefig('auc.png',bbox_inches="tight")

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)

accuracy=accuracy_score(predictions_SVM, y_test)*100
print(accuracy)
#a={"Accuracy": accuracy, "fpr": test_fpr,"tpr": test_tpr}
#b=a.tolist()
data = {'accuracy':accuracy,'fpr':test_fpr.tolist(),'tpr':test_tpr.tolist()}
print(data)

with open('Output/Accuracy.json', 'w') as f:
	json.dump(data,f, sort_keys=True, indent=4, separators=(',', ': '))
    	

#Accuracy=accuracy_score(predictions_SVM, y_test)*100
#a={"Accuracy": Accuracy, "fpr": test_fpr,"tpr": test_tpr}
#b=a.tolist()
#data = {'accuracy':accuracy,'fpr':test_fpr.tolist(),'tpr':test_tpr.tolist()}
'''
with open('Output/Accuracy.json', 'w') as f:
	json.dump(data,f, indent=4, separators=(',', ': '))

'''
with open('plots.json', 'w') as fd:
	json.dump(
        {
            "plots": [
                {"fpr": fp, "tpr": tp, "threshold": t}
                for fp, tp, t in zip(test_fpr.tolist(), test_tpr.tolist(), te_thresholds.tolist())
            ]
        },
        fd,
        indent=4,
    )   



#a=plt.plot_roc_curve(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))

#plt.legend()
#plt.xlabel("True Positive Rate")
#plt.ylabel("False Positive Rate")
#plt.title("AUC(ROC curve)")

plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.title("AUC(ROC curve)")
plt.rcParams["font.size"] = "22"

plot_confusion_matrix(SVM, x_test, y_test, normalize='true')

plt.savefig('confusion_mat.png',bbox_inches="tight")

'''
a=confusion_matrix(y_test,predictions_SVM)
b=a.tolist()
with open('Output/Confusion_matrix.json', 'w') as f:
    json.dump(b, f)
'''

