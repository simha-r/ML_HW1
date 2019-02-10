# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from data_loader import load_data

dataset = "mushrooms" # SET DATASET TO USE. "mushrooms" or "credit_cards"
data_x, data_y = load_data(dataset)

data_train_x, data_test_x, data_train_y, data_test_y = ms.train_test_split(data_x, data_y, test_size=0.3, random_state=0,stratify=data_y)

pipeM = Pipeline([ #('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=False,random_state=55))])

d = data_x.shape[1]
hiddens_data = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,3.01,1)]

params = {'MLP__activation':['relu','logistic'],'MLP__hidden_layer_sizes':hiddens_data,'MLP__alpha': alphas}

data_clf = basicResults(pipeM,data_train_x,data_train_y,data_test_x,data_test_y,params,'ANN',dataset)

data_final_params = data_clf.best_params_

pipeM.set_params(**data_final_params)
makeTimingCurve(data_x,data_y,pipeM,'ANN',dataset)

iterationLC(pipeM,data_train_x,data_train_y,data_test_x,data_test_y,{'MLP__max_iter':[2**x for x in range(8)]},'ANN',dataset=dataset)