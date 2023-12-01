#import modelcreation
import pickle
import numpy as np
from modelcreationfunctions import prepare as prep
import pandas as pd
#Quitar la siguientes (proximamente):
#from sklearn.base import BaseEstimator, ClassifierMixin
#from sklearn.decomposition import PCA
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import VotingClassifier
#from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

import chardet
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Triage

#Mistriage
def mistriage():
    with open("mistriage_model.pkl", "rb") as file:
        mistriagemodel=pickle.load(file)
    return mistriagemodel

#Explicit KTAS
def ktas():
    with open("ktas_model.pkl", "rb") as file:
        ktasmodel=pickle.load(file)
    return ktasmodel
"""
#Aquí abajo propongo un ejemplo para guiarse

input=[2,2,71,3,3,2,"right ocular pain",1,1,2,160,100,84,18,36.6,100,2,"Corneal abrasion",1,4,2,86,5,1]

#Para KTAS:
ktas_prep=prep(input,False,True,"data.csv")
pred=ktas().predict(ktas_prep)
ktas_pred=pred[0]
print("KTAS Prediction: "+str(ktas_pred))

#Para Mistriage:
mis_prep=prep(input,False,True,"data.csv")
pred=mistriage().predict(ktas_prep)
mis_pred=pred[0]
print("Mistriage Prediction: "+str(mis_pred))
"""




#Lung (esto lo vamos a mover de acá próximamente pero para no cambiar nada aquí lo dejo)

#Lung Questions
def load_lung_csv(path):
    #Lung Cancer CSV
    lungdf=pd.read_csv(path)
    lungdf=lungdf.replace(["M"],1)
    lungdf=lungdf.replace(["F"],2)
    lungdf=lungdf.replace(["YES"],2)
    lungdf=lungdf.replace(["NO"],1)

    #Data Matrix Construction
    y=np.array(lungdf["LUNG_CANCER"])
    lungdf=lungdf.drop("LUNG_CANCER",axis=1)
    X=lungdf.to_numpy()
    return X,y

def completedata(X,y):
    smote=SMOTE()
    return smote.fit_resample(X,y)
    



"""
def lung_symptoms_model(path="Datasets/LungCancerSurvey/survey lung cancer.csv"):
    #Importar datos 

    #Lung Cancer CSV
    lungdf=pd.read_csv(path)
    lungdf=lungdf.replace(["M"],1)
    lungdf=lungdf.replace(["F"],2)
    lungdf=lungdf.replace(["YES"],2)
    lungdf=lungdf.replace(["NO"],1)

    #Data Matrix Construction

    y=np.array(lungdf["LUNG_CANCER"])
    lungdf=lungdf.drop("LUNG_CANCER",axis=1)
    X=lungdf.to_numpy()

    #Data Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    #Random Forest
    randomforest = RandomForestClassifier(n_estimators=1000)
    randomforest.fit(X_train, Y_train)
    predictions = randomforest.predict(X_test)

    results = pd.DataFrame([preds, y_test], index=['Predicted', 'Observed']).T

    results['iseq'] = results.Observed.eq(results.Predicted).astype(int)
    correct = (results.iseq.sum() / len(results))
    print(f"Random Forest %Correct: {correct*100}%")
"""

class randomforest:
    """ Random Forest """
    def __init__(self,n_estimators):
        model=RandomForestClassifier(n_estimators)
        self.model=model
        self.n_estimators=n_estimators
        self.trainprobs=[]
        self.trainpreds=[]
        self.trainacc=[]
        self.probs=[]
        self.preds=[]

    def train(self,X,y,test_size):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        self.model.fit(Xtrain,ytrain)
        self.trainpreds=self.model.predict(Xtest)
        self.trainprobs=self.model.predict_proba(Xtest)
        total=0
        hits=0
        for i in range(0,len(ytest)-1):
            if ytest[i]==self.trainpreds[i]:
                hits+=1
            total+=1
        self.trainacc=hits/total

    def predict(self,X):
        self.preds=self.model.predict(X)
        self.probs=self.model.predict_proba(X)