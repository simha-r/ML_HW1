
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.svm import SVC
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from data_loader import load_data
from imblearn.pipeline import Pipeline as Pipeline1
from imblearn.over_sampling import SMOTE, RandomOverSampler



dataset = "diabetes" # SET DATASET TO USE. "mushrooms" or "credit_cards"
data_x, data_y = load_data(dataset)
data_train_x, data_test_x, data_train_y, data_test_y = ms.train_test_split(data_x, data_y, test_size=0.3, random_state=0,stratify=data_y)

pipeM = Pipeline1([ ('Scale',StandardScaler()),
                ('sampling', SMOTE()),
                ('SVM', SVC(random_state=0))])

N_data = data_train_x.shape[0]
params = {'SVM__kernel': ['linear', 'poly', 'rbf'], 'SVM__C': [.1,.5,1],'SVM__gamma': ['scale']}
complexity_params = {'name': 'SVM__C', 'display_name': 'Penalty', 'values': np.arange(0.001, 2.5, 0.1)}

data_clf = basicResults(pipeM,data_train_x,data_train_y,data_test_x,data_test_y,params,'SVM',dataset,scorer='f1',complexity_curve=True,complexity_params=complexity_params,clf_name='SVM')
data_final_params=data_clf.best_params_

pipeM.set_params(**data_final_params)
makeTimingCurve(data_x,data_y,pipeM,'SVM',dataset)

iterationLC(pipeM,data_train_x,data_train_y,data_test_x,data_test_y,{'SVM__max_iter':range(1,250,10)},'SVM',dataset=dataset,scorer='f1')