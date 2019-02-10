# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:55:38 2017

@author: JTay
"""
import numpy as np
from time import clock
import sklearn.model_selection as ms
import pandas as pd
from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix
from sklearn.utils import compute_sample_weight
from plotting import plot_confusion_matrix, plot_learning_curve, plot_model_timing, plot_model_complexity_curve
from sklearn import tree
from data_loader import export_decision_tree
from sklearn.model_selection import validation_curve


OUTPUT_DIRECTORY = 'output'
def balanced_accuracy(truth,pred):
    # wts = compute_sample_weight('balanced',truth)
    # return accuracy_score(truth,pred,sample_weight=wts)
    return accuracy_score(truth,pred)

# scorer = make_scorer(balanced_accuracy)
# scorer = 'f1'
    
def basicResults(clfObj,trgX,trgY,tstX,tstY,params,clf_type=None,dataset=None,feature_names=None,scorer='accuracy',complexity_curve=False,complexity_params=None,clf_name=""):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    print("Starting grid search--------")
    cv = ms.GridSearchCV(clfObj,n_jobs=1,param_grid=params,refit=True,verbose=10,cv=5,scoring=scorer)
    cv.fit(trgX,trgY)

    # export_decision_tree(cv, feature_names, dataset)

    print("Ended     grid search--------")
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/{}_{}_reg.csv'.format(clf_type,dataset),index=False)
    test_score = cv.score(tstX,tstY)

    test_y_predicted = cv.predict(tstX)


    # PLOT Confusion Matrix
    cnf_matrix = confusion_matrix(tstY, test_y_predicted)
    plt = plot_confusion_matrix(cnf_matrix,title='Confusion Matrix: {} - {}'.format(clf_type, dataset))
    OUTPUT_DIRECTORY = "output"
    plt.savefig('{}/images/{}_{}_CM.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150,
                bbox_inches='tight')


    with open('./output/test results.csv','a') as f:
        f.write('{},{},{},{}\n'.format(clf_type,dataset,test_score,cv.best_params_))    
    N = trgY.shape[0]    

    # Plot Learning Curve
    # curve = ms.learning_curve(cv.best_estimator_,trgX,trgY,cv=3,train_sizes=np.linspace(0.1, 1.0, 20),verbose=10,scoring=scorer)
    curve = ms.learning_curve(cv.best_estimator_, trgX, trgY, cv=3, train_sizes=np.linspace(0.2, 1.0, 10), verbose=10,
                              scoring=scorer)
    curve_train_scores = pd.DataFrame(index = curve[0],data = curve[1])
    curve_test_scores  = pd.DataFrame(index = curve[0],data = curve[2])
    curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type,dataset))
    curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type,dataset))

    plt = plot_learning_curve('Learning Curve: {} - {}'.format(clf_type, dataset),curve[0],curve[1], curve[2],y_label=scorer)
    plt.savefig('{}/images/{}_{}_LC.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150)


    if complexity_curve:
        make_complexity_curve(trgX, trgY, complexity_params['name'], complexity_params['display_name'],
                              complexity_params['values'], clfObj,clf_name=clf_name,dataset=dataset,dataset_readable_name=dataset)
        print("Drew complexity curve")


    return cv

    
# def iterationLC(clfObj,trgX,trgY,tstX,tstY,params,clf_type=None,dataset=None,dataset_readable_name=None):
def iterationLC(clfObj,trgX,trgY,tstX,tstY,params,clf_type=None,dataset=None,
                 dataset_readable_name=None, balanced_dataset=False, x_scale='linear', seed=55, threads=1,scorer='accuracy'):
    if not dataset_readable_name:
        dataset_readable_name = dataset

    np.random.seed(50)
    if clf_type is None or dataset is None:
        print("clf_type = ",clf_type)
        print("dataset = ", dataset)
        raise
    cv = ms.GridSearchCV(clfObj,n_jobs=1,param_grid=params,refit=True,verbose=10,cv=5,scoring=scorer)
    cv.fit(trgX,trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/ITER_base_{}_{}.csv'.format(clf_type,dataset),index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:        
        d['param_{}'.format(name)].append(value)
        clfObj.set_params(**{name:value})
        clfObj.fit(trgX,trgY)
        pred = clfObj.predict(trgX)
        d['train acc'].append(balanced_accuracy(trgY,pred))
        clfObj.fit(trgX,trgY)
        pred = clfObj.predict(tstX)
        d['test acc'].append(balanced_accuracy(tstY,pred))
        print(value)
    d = pd.DataFrame(d)
    d.to_csv('./output/ITERtestSET_{}_{}.csv'.format(clf_type,dataset),index=False)

    plt = plot_learning_curve('{} - {} ({})'.format(clf_type, dataset_readable_name, name),
                              d['param_{}'.format(name)], d['train acc'], d['test acc'],
                              multiple_runs=False, x_scale=x_scale,
                              x_label='Value',y_label=scorer)
    plt.savefig('{}/images/{}_{}_ITER_LC.png'.format(OUTPUT_DIRECTORY, clf_type, dataset), format='png', dpi=150)


    return cv    
    
def add_noise(y,frac=0.1):
    np.random.seed(456)
    n = y.shape[0]
    sz = int(n*frac)
    ind = np.random.choice(np.arange(n),size=sz,replace=False)
    tmp = y.copy()
    tmp[ind] = 1-tmp[ind]
    return tmp


def makeTimingCurve(x, y, clf, clf_name, dataset, dataset_readable_name=None, verbose=False, seed=42):
    if not dataset_readable_name:
        dataset_readable_name = dataset
    # np.linspace(0.1, 1, num=10)  #

    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tests = 5
    out = dict()
    out['train'] = np.zeros(shape=(len(sizes), tests))
    out['test'] = np.zeros(shape=(len(sizes), tests))
    for i, frac in enumerate(sizes):
        for j in range(tests):
            np.random.seed(seed)
            x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=1 - frac, random_state=seed)
            st = clock()
            clf.fit(x_train, y_train)
            out['train'][i, j] = (clock() - st)
            st = clock()
            clf.predict(x_test)
            out['test'][i, j] = (clock() - st)

    train_df = pd.DataFrame(out['train'], index=sizes)
    test_df = pd.DataFrame(out['test'], index=sizes)
    plt = plot_model_timing('{} - {}'.format(clf_name, dataset_readable_name),
                            np.array(sizes) * 100, train_df, test_df)
    plt.savefig('{}/images/{}_{}_TC.png'.format(OUTPUT_DIRECTORY, clf_name, dataset), format='png', dpi=150)

    out = pd.DataFrame(index=sizes)
    out['train'] = np.mean(train_df, axis=1)
    out['test'] = np.mean(test_df, axis=1)
    out.to_csv('{}/{}_{}_timing.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset))


def make_complexity_curve(x, y, param_name, param_display_name, param_values, clf, clf_name, dataset,
                          dataset_readable_name, x_scale='linear', verbose=False,threads=1, scorer='accuracy'):
    curr_scorer = scorer

    train_scores, test_scores = validation_curve(clf, x, y, param_name, param_values, cv=5, verbose=verbose,
                                                 scoring=curr_scorer, n_jobs=threads)

    curve_train_scores = pd.DataFrame(index=param_values, data=train_scores)
    curve_test_scores = pd.DataFrame(index=param_values, data=test_scores)
    curve_train_scores.to_csv('{}/{}_{}_{}_MC_train.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset, param_name))
    curve_test_scores.to_csv('{}/{}_{}_{}_MC_test.csv'.format(OUTPUT_DIRECTORY, clf_name, dataset, param_name))
    plt = plot_model_complexity_curve(
        'Model Complexity: {} - {} ({})'.format(clf_name, dataset_readable_name, param_display_name),
        param_values,train_scores, test_scores, x_scale=x_scale,x_label=param_display_name)
    plt.savefig('{}/images/{}_{}_{}_MC.png'.format(OUTPUT_DIRECTORY, clf_name, dataset, param_name), format='png',
                dpi=150)