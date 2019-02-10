# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as dtclf
import pandas as pd
from helpers import basicResults, makeTimingCurve, iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_loader import load_data
from sklearn.linear_model import SGDClassifier

from imblearn.pipeline import Pipeline as Pipeline1
from imblearn.over_sampling import SMOTE, RandomOverSampler



dataset = "diabetes" # SET DATASET TO USE. "mushrooms" or "credit_cards"
data_x, data_y = load_data(dataset)
data_train_x, data_test_x, data_train_y, data_test_y = ms.train_test_split(data_x, data_y, test_size=0.3,
                                                                           random_state=0, stratify=data_y)

base = dtclf(max_depth=3)

alphas = [-1, -1e-3, -(1e-3) * 10 ** -0.5, -1e-2, -(1e-2) * 10 ** -0.5, -1e-1, -(1e-1) * 10 ** -0.5, 0,
          (1e-1) * 10 ** -0.5, 1e-1, (1e-2) * 10 ** -0.5, 1e-2, (1e-3) * 10 ** -0.5, 1e-3]

params = {'Boost__n_estimators': [20, 40, 60, 80, 100, 120, 140,160,180,200,250,300,400],
          'Boost__learning_rate': [(2 ** x) / 100 for x in range(8)] + [1]}
# paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
#           'Boost__base_estimator__alpha':alphas}


booster = AdaBoostClassifier(algorithm='SAMME', learning_rate=1, base_estimator=base, random_state=55)

pipeM = Pipeline1([('Scale',StandardScaler()),
    ('sampling', SMOTE()),
    ('Boost', booster)])

data_clf = basicResults(pipeM, data_train_x, data_train_y, data_test_x, data_test_y, params, 'Boost', dataset, scorer='f1')
data_final_params = data_clf.best_params_

pipeM.set_params(**data_final_params)
makeTimingCurve(data_x, data_y, pipeM, 'Boost', dataset)

# iterationLC(pipeA,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','adult')