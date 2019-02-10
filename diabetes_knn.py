# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from data_loader import load_data
from imblearn.pipeline import Pipeline as Pipeline1
from imblearn.over_sampling import SMOTE, RandomOverSampler


dataset = "diabetes" # SET DATASET TO USE. "mushrooms" or "credit_cards"
data_x, data_y = load_data(dataset)

data_train_x, data_test_x, data_train_y, data_test_y = ms.train_test_split(data_x, data_y, test_size=0.3, random_state=0,stratify=data_y)

pipeM = Pipeline1([('Scale',StandardScaler()),
                ('sampling', SMOTE()),
                ('KNN', knnC())])


params = {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,5),'KNN__weights':['uniform','distance']}

data_clf = basicResults(pipeM,data_train_x,data_train_y,data_test_x,data_test_y,params,'KNN',dataset,scorer='f1')
data_final_params=data_clf.best_params_

pipeM.set_params(**data_final_params)
makeTimingCurve(data_x,data_y,pipeM,'KNN',dataset)