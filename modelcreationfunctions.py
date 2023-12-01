import numpy as np
import pandas as pd

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

import pickle


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


#Triage

#Triage: Auxiliary Functions

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names

def one_hot_encoder(dataframe, categorical_columns, nan_as_category=False):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns,
                               dummy_na=nan_as_category, drop_first=True)
    new_columns = [col for col in dataframe.columns if col not in original_columns]
    return dataframe, new_columns

def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.05)
    quartile3 = variable.quantile(0.95)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)

def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na

def load_triage_data(path,mistriage,ktas):
    rawdata = open(path, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    df = pd.read_csv(path, encoding=encoding, sep=";")
    df['NRS_pain'] = df[["NRS_pain"]].replace("#BOÞ!",np.NaN)
    df["KTAS duration_min"]=df[["KTAS duration_min"]].replace(regex={",":"."})
    df['KTAS duration_min']=df['KTAS duration_min'].astype(dtype=float)

    df["NRS_pain"] = df.groupby(["mistriage", "KTAS_expert"])["NRS_pain"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Saturation"] = df.groupby(["mistriage", "KTAS_expert"])["Saturation"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Diagnosis in ED"] = df.groupby(["mistriage", "KTAS_expert"])["Diagnosis in ED"].transform(lambda x: x.fillna(x.mode()[0]))

    injury_cat = ['No','Yes']
    sex_cat = ['Female','Male']
    pain_cat = ['No','Yes']
    mental_cat = ['Alert','Verbose Response','Pain Response','Unresponsive']
    group_cat = ['Local ED (3th Degree)', 'Regional ED (4th Degree)']
    arrival_mode_cat = ['Walking','Public Ambulance', 'Private Vehicle','Private Ambulance', 'Other','Other','Other']
    disposition_cat = ['Discharge', 'Admission to Ward', 'Admission to ICU', 'Discharge', 'Transfer', 'Death', 'Surgery']
    KTAS_cat = ['Emergency','Emergency','Emergency', 'Non-Emergency', 'Non-Emergency']
    mistriage_cat = ['Normal Triage','Over Triage','Under Triage']
    df['NRS_pain'] = df['NRS_pain'].apply(lambda x:int(x))

    df.loc[df["Sex"] == 1, "Sex"] = sex_cat[0]
    df.loc[df["Sex"] == 2, "Sex"] = sex_cat[1]

    df.loc[df["Injury"] == 1, "Injury"] = injury_cat[0]
    df.loc[df["Injury"] == 2, "Injury"] = injury_cat[1]

    df.loc[df["Pain"] == 0, "Pain"] = pain_cat[0]
    df.loc[df["Pain"] == 1, "Pain"] = pain_cat[1]

    df.loc[df["Mental"] == 1, "Mental"] = mental_cat[0]
    df.loc[df["Mental"] == 2, "Mental"] = mental_cat[1]
    df.loc[df["Mental"] == 3, "Mental"] = mental_cat[2]
    df.loc[df["Mental"] == 4, "Mental"] = mental_cat[3]

    df.loc[df["Group"] == 1, "Group"] = group_cat[0]
    df.loc[df["Group"] == 2, "Group"] = group_cat[1]

    df.loc[df["Arrival mode"] == 1, "Arrival mode"] = arrival_mode_cat[0]
    df.loc[df["Arrival mode"] == 2, "Arrival mode"] = arrival_mode_cat[1]
    df.loc[df["Arrival mode"] == 3, "Arrival mode"] = arrival_mode_cat[2]
    df.loc[df["Arrival mode"] == 4, "Arrival mode"] = arrival_mode_cat[3]
    df.loc[df["Arrival mode"] == 5, "Arrival mode"] = arrival_mode_cat[4]
    df.loc[df["Arrival mode"] == 6, "Arrival mode"] = arrival_mode_cat[5]
    df.loc[df["Arrival mode"] == 7, "Arrival mode"] = arrival_mode_cat[6]

    df.loc[df["Disposition"] == 1, "Disposition"] = disposition_cat[0]
    df.loc[df["Disposition"] == 2, "Disposition"] = disposition_cat[1]
    df.loc[df["Disposition"] == 3, "Disposition"] = disposition_cat[2]
    df.loc[df["Disposition"] == 4, "Disposition"] = disposition_cat[3]
    df.loc[df["Disposition"] == 5, "Disposition"] = disposition_cat[4]
    df.loc[df["Disposition"] == 6, "Disposition"] = disposition_cat[5]
    df.loc[df["Disposition"] == 7, "Disposition"] = disposition_cat[6]


    df.loc[df["KTAS_RN"] == 1, "KTAS_RN"] = KTAS_cat[0]
    df.loc[df["KTAS_RN"] == 2, "KTAS_RN"] = KTAS_cat[1]
    df.loc[df["KTAS_RN"] == 3, "KTAS_RN"] = KTAS_cat[2]
    df.loc[df["KTAS_RN"] == 4, "KTAS_RN"] = KTAS_cat[3]
    df.loc[df["KTAS_RN"] == 5, "KTAS_RN"] = KTAS_cat[4]

    if mistriage:
        df.loc[df["KTAS_expert"] == 1, "KTAS_expert"] = KTAS_cat[0]
        df.loc[df["KTAS_expert"] == 2, "KTAS_expert"] = KTAS_cat[1]
        df.loc[df["KTAS_expert"] == 3, "KTAS_expert"] = KTAS_cat[2]
        df.loc[df["KTAS_expert"] == 4, "KTAS_expert"] = KTAS_cat[3]
        df.loc[df["KTAS_expert"] == 5, "KTAS_expert"] = KTAS_cat[4]

    df.loc[df["mistriage"] == 0, "mistriage"] = mistriage_cat[0]
    df.loc[df["mistriage"] == 1, "mistriage"] = mistriage_cat[1]
    df.loc[df["mistriage"] == 2, "mistriage"] = mistriage_cat[2]

    df[["SBP","DBP","HR","RR","BT","Saturation"]] = df[["SBP","DBP","HR","RR","BT","Saturation"]].replace("??",0).astype(str).astype(float)
    df['SBP'] = df['SBP'].replace(0,df['SBP'].mode()[0])
    df['DBP'] = df['DBP'].replace(0,df['DBP'].mode()[0])

    interval = (0, 25, 45, 60, 100)
    cats = ['Young', 'Adult', 'Mid_Age', 'Old']
    df["New_Age"] = pd.cut(df.Age, interval, labels=cats)

    df.loc[df['SBP'] < 80, 'New_SBP'] = 'Low'
    df.loc[(df["SBP"] >= 80) & (df["SBP"] <= 120), 'New_SBP'] = 'Normal'
    df.loc[df['SBP'] > 120, 'New_SBP'] = 'High'

    df.loc[df['DBP'] < 60, 'New_DBP'] = 'Low'
    df.loc[(df["DBP"] >= 60) & (df["DBP"] <= 80), 'New_DBP'] = 'Normal'
    df.loc[df['DBP'] > 80, 'New_DBP'] = 'High'

    df.loc[df['HR'] < 45, 'New_HR'] = 'Low'
    df.loc[(df["HR"] >= 45) & (df["HR"] <= 100), 'New_HR'] = 'Normal'
    df.loc[df['HR'] > 100, 'New_HR'] = 'High'

    df.loc[df['RR'] < 12, 'New_RR'] = 'Low'
    df.loc[(df["RR"] >= 12) & (df["RR"] <= 25), 'New_RR'] = 'Normal'
    df.loc[df['RR'] > 25, 'New_RR'] = 'High'

    df.loc[df['BT'] < 36.4, 'New_BT'] = 'Low'
    df.loc[(df["BT"] >= 36.4) & (df["BT"] <= 37.6), 'New_BT'] = 'Normal'
    df.loc[df['BT'] > 37.6, 'New_BT'] = 'High'

    df.loc[df['NRS_pain'] < 3, 'New_NRS_pain'] = 'Low Pain'
    df.loc[(df["NRS_pain"] >= 3) & (df["NRS_pain"] <= 7), 'New_NRS_pain'] = 'Pain'
    df.loc[df['NRS_pain'] > 7, 'New_NRS_pain'] = 'High Pain'

    df.loc[df['KTAS duration_min'] < 10, 'New_KTAS_duration_min'] = 'Immediate'
    df.loc[(df["KTAS duration_min"] >= 10) & (df["KTAS duration_min"] <= 60), 'New_KTAS_duration_min'] = 'Very Urgent'
    df.loc[(df["KTAS duration_min"] >= 61) & (df["KTAS duration_min"] <= 120), 'New_KTAS_duration_min'] = 'Urgent'
    df.loc[(df["KTAS duration_min"] >= 121) & (df["KTAS duration_min"] <= 240), 'New_KTAS_duration_min'] = 'Standart'
    df.loc[df['KTAS duration_min'] > 240, 'New_KTAS_duration_min'] = 'Non-Urgent'

    df.loc[df['Length of stay_min'] < 10, 'New_Length_of_stay_min'] = 'Immediate'
    df.loc[(df["Length of stay_min"] >= 10) & (df["Length of stay_min"] <= 60), 'New_Length_of_stay_min'] = 'Very Urgent'
    df.loc[(df["Length of stay_min"] >= 61) & (df["Length of stay_min"] <= 120), 'New_Length_of_stay_min'] = 'Urgent'
    df.loc[(df["Length of stay_min"] >= 121) & (df["Length of stay_min"] <= 240), 'New_Length_of_stay_min'] = 'Standart'
    df.loc[df['Length of stay_min'] > 240, 'New_Length_of_stay_min'] = 'Non-Urgent'

    outlier_column = ["Length of stay_min","Saturation","RR","BT"]
    for i in outlier_column:
        replace_with_thresholds(df,i)

    new_df = df[["Group", "Sex", "Patients number per hour", "Arrival mode", "Injury", "Mental", "Pain", "Saturation", "KTAS_RN",
    "Disposition", "KTAS_expert", "Length of stay_min", "mistriage", "New_Age", "New_SBP", "New_DBP", "New_HR",
    "New_RR", "New_BT", "New_NRS_pain", "New_KTAS_duration_min", "New_Length_of_stay_min"]]
    new_df.head()
    if mistriage:
        like_num = [col for col in new_df.columns if new_df[col].dtypes != 'O' and len(new_df[col].value_counts()) > 10]
        cols_need_scale = [col for col in new_df.columns if col not in like_num
                        and col not in "mistriage KTAS_expert"]
    elif ktas:
        like_num = [col for col in new_df.columns if new_df[col].dtypes != 'O' and len(new_df[col].value_counts()) > 10]
        cols_need_scale = [col for col in new_df.columns if col not in like_num
                        and col not in 'mistriage KTAS_expert']
    for col in like_num:
        new_df[col] = robust_scaler(new_df[col])
    new_df, one_hot_encodeds = one_hot_encoder(new_df, cols_need_scale)
    return new_df

def prepare(test2,mistriage,ktas,path):
    rawdata = open(path, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']

    df2 = pd.read_csv(path, encoding=encoding, sep=";")
    df2.loc[0]=test2
    df=df2

    df['NRS_pain'] = df[["NRS_pain"]].replace("#BOÞ!",np.NaN)
    df["KTAS duration_min"]=df[["KTAS duration_min"]].replace(regex={",":"."})
    df['KTAS duration_min']=df['KTAS duration_min'].astype(dtype=float)

    df["NRS_pain"] = df.groupby(["mistriage", "KTAS_expert"])["NRS_pain"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Saturation"] = df.groupby(["mistriage", "KTAS_expert"])["Saturation"].transform(lambda x: x.fillna(x.mode()[0]))
    df["Diagnosis in ED"] = df.groupby(["mistriage", "KTAS_expert"])["Diagnosis in ED"].transform(lambda x: x.fillna(x.mode()[0]))



    injury_cat = ['No','Yes']
    sex_cat = ['Female','Male']
    pain_cat = ['No','Yes']
    mental_cat = ['Alert','Verbose Response','Pain Response','Unresponsive']
    group_cat = ['Local ED (3th Degree)', 'Regional ED (4th Degree)']
    arrival_mode_cat = ['Walking','Public Ambulance', 'Private Vehicle','Private Ambulance', 'Other','Other','Other']
    disposition_cat = ['Discharge', 'Admission to Ward', 'Admission to ICU', 'Discharge', 'Transfer', 'Death', 'Surgery']
    KTAS_cat = ['Emergency','Emergency','Emergency', 'Non-Emergency', 'Non-Emergency']
    mistriage_cat = ['Normal Triage','Over Triage','Under Triage']

    df['NRS_pain'] = df['NRS_pain'].apply(lambda x:int(x))

    df.loc[df["Sex"] == 1, "Sex"] = sex_cat[0]
    df.loc[df["Sex"] == 2, "Sex"] = sex_cat[1]

    df.loc[df["Injury"] == 1, "Injury"] = injury_cat[0]
    df.loc[df["Injury"] == 2, "Injury"] = injury_cat[1]

    df.loc[df["Pain"] == 0, "Pain"] = pain_cat[0]
    df.loc[df["Pain"] == 1, "Pain"] = pain_cat[1]

    df.loc[df["Mental"] == 1, "Mental"] = mental_cat[0]
    df.loc[df["Mental"] == 2, "Mental"] = mental_cat[1]
    df.loc[df["Mental"] == 3, "Mental"] = mental_cat[2]
    df.loc[df["Mental"] == 4, "Mental"] = mental_cat[3]

    df.loc[df["Group"] == 1, "Group"] = group_cat[0]
    df.loc[df["Group"] == 2, "Group"] = group_cat[1]

    df.loc[df["Arrival mode"] == 1, "Arrival mode"] = arrival_mode_cat[0]
    df.loc[df["Arrival mode"] == 2, "Arrival mode"] = arrival_mode_cat[1]
    df.loc[df["Arrival mode"] == 3, "Arrival mode"] = arrival_mode_cat[2]
    df.loc[df["Arrival mode"] == 4, "Arrival mode"] = arrival_mode_cat[3]
    df.loc[df["Arrival mode"] == 5, "Arrival mode"] = arrival_mode_cat[4]
    df.loc[df["Arrival mode"] == 6, "Arrival mode"] = arrival_mode_cat[5]
    df.loc[df["Arrival mode"] == 7, "Arrival mode"] = arrival_mode_cat[6]

    df.loc[df["Disposition"] == 1, "Disposition"] = disposition_cat[0]
    df.loc[df["Disposition"] == 2, "Disposition"] = disposition_cat[1]
    df.loc[df["Disposition"] == 3, "Disposition"] = disposition_cat[2]
    df.loc[df["Disposition"] == 4, "Disposition"] = disposition_cat[3]
    df.loc[df["Disposition"] == 5, "Disposition"] = disposition_cat[4]
    df.loc[df["Disposition"] == 6, "Disposition"] = disposition_cat[5]
    df.loc[df["Disposition"] == 7, "Disposition"] = disposition_cat[6]


    df.loc[df["KTAS_RN"] == 1, "KTAS_RN"] = KTAS_cat[0]
    df.loc[df["KTAS_RN"] == 2, "KTAS_RN"] = KTAS_cat[1]
    df.loc[df["KTAS_RN"] == 3, "KTAS_RN"] = KTAS_cat[2]
    df.loc[df["KTAS_RN"] == 4, "KTAS_RN"] = KTAS_cat[3]
    df.loc[df["KTAS_RN"] == 5, "KTAS_RN"] = KTAS_cat[4]

    df.loc[df["KTAS_expert"] == 1, "KTAS_expert"] = KTAS_cat[0]
    df.loc[df["KTAS_expert"] == 2, "KTAS_expert"] = KTAS_cat[1]
    df.loc[df["KTAS_expert"] == 3, "KTAS_expert"] = KTAS_cat[2]
    df.loc[df["KTAS_expert"] == 4, "KTAS_expert"] = KTAS_cat[3]
    df.loc[df["KTAS_expert"] == 5, "KTAS_expert"] = KTAS_cat[4]

    df.loc[df["mistriage"] == 0, "mistriage"] = mistriage_cat[0]
    df.loc[df["mistriage"] == 1, "mistriage"] = mistriage_cat[1]
    df.loc[df["mistriage"] == 2, "mistriage"] = mistriage_cat[2]

    df[["SBP","DBP","HR","RR","BT","Saturation"]] = df[["SBP","DBP","HR","RR","BT","Saturation"]].replace("??",0).astype(str).astype(float)
    df['SBP'] = df['SBP'].replace(0,df['SBP'].mode()[0])
    df['DBP'] = df['DBP'].replace(0,df['DBP'].mode()[0])

    interval = (0, 25, 45, 60, 100)
    cats = ['Young', 'Adult', 'Mid_Age', 'Old']
    df["New_Age"] = pd.cut(df.Age, interval, labels=cats)

    df.loc[df['SBP'] < 80, 'New_SBP'] = 'Low'
    df.loc[(df["SBP"] >= 80) & (df["SBP"] <= 120), 'New_SBP'] = 'Normal'
    df.loc[df['SBP'] > 120, 'New_SBP'] = 'High'

    df.loc[df['DBP'] < 60, 'New_DBP'] = 'Low'
    df.loc[(df["DBP"] >= 60) & (df["DBP"] <= 80), 'New_DBP'] = 'Normal'
    df.loc[df['DBP'] > 80, 'New_DBP'] = 'High'

    df.loc[df['HR'] < 45, 'New_HR'] = 'Low'
    df.loc[(df["HR"] >= 45) & (df["HR"] <= 100), 'New_HR'] = 'Normal'
    df.loc[df['HR'] > 100, 'New_HR'] = 'High'

    df.loc[df['RR'] < 12, 'New_RR'] = 'Low'
    df.loc[(df["RR"] >= 12) & (df["RR"] <= 25), 'New_RR'] = 'Normal'
    df.loc[df['RR'] > 25, 'New_RR'] = 'High'

    df.loc[df['BT'] < 36.4, 'New_BT'] = 'Low'
    df.loc[(df["BT"] >= 36.4) & (df["BT"] <= 37.6), 'New_BT'] = 'Normal'
    df.loc[df['BT'] > 37.6, 'New_BT'] = 'High'

    df.loc[df['NRS_pain'] < 3, 'New_NRS_pain'] = 'Low Pain'
    df.loc[(df["NRS_pain"] >= 3) & (df["NRS_pain"] <= 7), 'New_NRS_pain'] = 'Pain'
    df.loc[df['NRS_pain'] > 7, 'New_NRS_pain'] = 'High Pain'

    df.loc[df['KTAS duration_min'] < 10, 'New_KTAS_duration_min'] = 'Immediate'
    df.loc[(df["KTAS duration_min"] >= 10) & (df["KTAS duration_min"] <= 60), 'New_KTAS_duration_min'] = 'Very Urgent'
    df.loc[(df["KTAS duration_min"] >= 61) & (df["KTAS duration_min"] <= 120), 'New_KTAS_duration_min'] = 'Urgent'
    df.loc[(df["KTAS duration_min"] >= 121) & (df["KTAS duration_min"] <= 240), 'New_KTAS_duration_min'] = 'Standart'
    df.loc[df['KTAS duration_min'] > 240, 'New_KTAS_duration_min'] = 'Non-Urgent'

    df.loc[df['Length of stay_min'] < 10, 'New_Length_of_stay_min'] = 'Immediate'
    df.loc[(df["Length of stay_min"] >= 10) & (df["Length of stay_min"] <= 60), 'New_Length_of_stay_min'] = 'Very Urgent'
    df.loc[(df["Length of stay_min"] >= 61) & (df["Length of stay_min"] <= 120), 'New_Length_of_stay_min'] = 'Urgent'
    df.loc[(df["Length of stay_min"] >= 121) & (df["Length of stay_min"] <= 240), 'New_Length_of_stay_min'] = 'Standart'
    df.loc[df['Length of stay_min'] > 240, 'New_Length_of_stay_min'] = 'Non-Urgent'
    outlier_column = ["Length of stay_min","Saturation","RR","BT"]
    for i in outlier_column:
        replace_with_thresholds(df,i)
    new_df = df[["Group", "Sex", "Patients number per hour", "Arrival mode", "Injury", "Mental", "Pain", "Saturation", "KTAS_RN",
    "Disposition", "KTAS_expert", "Length of stay_min", "mistriage", "New_Age", "New_SBP", "New_DBP", "New_HR",
    "New_RR", "New_BT", "New_NRS_pain", "New_KTAS_duration_min", "New_Length_of_stay_min"]]
    new_df.head()

    like_num = [col for col in new_df.columns if new_df[col].dtypes != 'O' and len(new_df[col].value_counts()) > 10]
    cols_need_scale = [col for col in new_df.columns if col not in like_num
                    and col not in 'mistriage KTAS_expert']

    for col in like_num:
        new_df[col] = robust_scaler(new_df[col])
    new_df, one_hot_encodeds = one_hot_encoder(new_df, cols_need_scale)
    new_df=new_df.drop(["KTAS_expert","mistriage"],axis=1)
    return new_df.head(1)

#Triage: Mistriage

def mistriage_model_creation(path):
    new_df=load_triage_data(path,True,False)
    X = new_df.drop(['mistriage',"KTAS_expert"], axis=1)
    y = np.ravel(new_df[['mistriage']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 357)
    rf_params = {"max_depth": [3, 5, 8],
             "max_features": [8, 15, 25],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10]}
    rf = RandomForestClassifier(random_state=357)
    gs_cv_rf = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
    rf_tuned=RandomForestClassifier(**gs_cv_rf.best_params_, random_state=357).fit(X_train,y_train)
    pkl_filename="mistriage_model.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(rf_tuned,file)

#Triage: Explicit KTAS Score

def ktas_model_creation(path):
    new_df=load_triage_data(path,False,True)
    X = new_df.drop(['mistriage',"KTAS_expert"], axis=1)
    y = np.ravel(new_df[['KTAS_expert']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 357)
    rf_params = {"max_depth": [3, 5, 8],
             "max_features": [8, 15, 25],
             "n_estimators": [200, 500, 1000],
             "min_samples_split": [2, 5, 10]}
    rf = RandomForestClassifier(random_state=357)
    gs_cv_rf = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
    rf_tuned=RandomForestClassifier(**gs_cv_rf.best_params_, random_state=357).fit(X_train,y_train)
    pkl_filename="ktas_model.pkl"
    with open(pkl_filename, "wb") as file:
        pickle.dump(rf_tuned,file)









































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

