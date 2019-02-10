# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:55:32 2017

Script for full tests, decision tree (pruned)

"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier as dtclf
from data_loader import load_data, export_decision_tree
from sklearn import tree
from imblearn.pipeline import Pipeline as Pipeline1
from imblearn.over_sampling import SMOTE, RandomOverSampler


dataset = "mushrooms" # SET DATASET TO USE. "mushrooms" or "credit_cards"
data_x, data_y,cols = load_data(dataset,return_cols=True)


data_train_x, data_test_x, data_train_y, data_test_y = ms.train_test_split(data_x, data_y, test_size=0.3, stratify=data_y)#, random_state=845)

pipeM = Pipeline([('DT',dtclf())])
label = "Poisonous"

max_depths = range(2,30)
params = {'DT__criterion':['gini','entropy'],'DT__class_weight':['balanced'],'DT__max_depth': max_depths}
complexity_params = {'name': 'DT__max_depth', 'display_name': 'Max Depth', 'values': max_depths}

print("Going to get basic results")
data_clf = basicResults(pipeM,data_train_x,data_train_y,data_test_x,data_test_y,params,'DT',dataset,feature_names=cols,complexity_curve=True,complexity_params=complexity_params,clf_name="DT")

data_final_params = data_clf.best_params_
best_dtree = data_clf.best_estimator_
# export_decision_tree(best_dtree,cols,dataset,label)

pipeM.set_params(**data_final_params)
makeTimingCurve(data_x,data_y,pipeM,'DT',dataset)






