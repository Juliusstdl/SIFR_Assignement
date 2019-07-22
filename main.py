#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# @author:   Julius Steidl
# date:     22.07.2019
# version:  1.5
# NOTE:     folder with .csv files is required:  ./Kickstarter_2019-07-18T03_20_05_009Z/
#           source:  https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2019-07-18T03_20_05_009Z.zip


from DatasetImport import DatasetImport
from Statistics import Statistics
from Preprocessing import Preprocessing
from Estimate import Estimate

import sys
import pandas as pd
import numpy as np
import ast
import re
import operator
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
#from sklearn.ensemble.voting_classifier import VotingClassifier
#from sklearn.mixture import DPGMM
#from sklearn.mixture import GMM
#from sklearn.mixture import GaussianMixture
#from sklearn.mixture import VBGMM

#import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None  # default='warn'


def rank_feature_importance(importances):
    importances.loc['Total',:]= importances.sum(axis=0)

    importance_ranking = (importances.iloc[-1:].reset_index(drop=True)).to_dict()
    for feature, value in importance_ranking.items():
        importance_ranking[feature] = value.get(0)

    importance_ranking = sorted(importance_ranking.items(), key=operator.itemgetter(1), reverse=True)

    # Create dataframe from dictionry with results:
    importance_ranking_df = pd.DataFrame.from_dict(importance_ranking)
    importance_ranking_df.columns = ['Feature', 'Importance']
    importance_ranking_df.sort_values(by=['Importance'])

    return importance_ranking_df


def main():

    # I. Importing Dataset & Data Cleaning:

    # Skipping new dataset import and statistics generation:
    skipimport = False; skipstats = False

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg in ['-skipall', 'skipall', '-skip', 'skip']:
                skipimport = True; skipstats = True
            elif arg in ['-skipimport', 'skipimport']:
                skipimport = True
            elif arg in ['-skipstats', 'skipstats', '-skipstatistics', 'skipstatistics']:
                skipstats = True


    path = 'Kickstarter_2019-07-18T03_20_05_009Z'  # date: 2019-07-18

    # saving / loading dataframe as .pickle file  (-> not necessary to parse .csv-files every time):
    if skipimport and os.path.exists(path+'.pickle') and os.path.isfile(path+'.pickle'):
        data = pd.read_pickle('./'+path+'.pickle')
    else:
        datasetImport = DatasetImport(path)
        data = datasetImport.data
        data.to_pickle(path+'.pickle')



    # II. Exploratory data analysis:

    # Generating some statistics for getting insight into dataset & for supporting manual feature-selection decisions:
    # NOTE:  Function call is optional
    if not skipstats:
        statistics = Statistics(data)
        statistics.generate_statistics(data)


    # Printing dataframe header (= all imported features & labels):
    print('\n============================= DATAFRAME HEADER ==============================')
    print(data.iloc[0]) # =header



    # III. Further Data Cleaning & Encoding:

    # Filtering out entries (=campaigns) with 'canceled' or 'live' state values (as label).
    # => Just using 'successful' and 'failed' campaigns for estimation:
    data_filtered = data.loc[data['state'].isin(['successful','failed'])]

    # Encoding string-value columns to logical datatypes (binary & ordinal) for estimation:
    preprocessing = Preprocessing(data_filtered)
    data_encoded = preprocessing.data_encoded



    # IV. Feature-Selection:

    # a) Manual Feature-Pre-Selection:

    # NOTE:  manually add/remove features in following line forfeature-selection:
    feature_preselection = ['backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'goal_exceeded']#, 'divergence'] #'spotlight'
    # features = ['backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence', 'goal_exceeded']

    X = data_encoded[feature_preselection]
    y = data_encoded['state']

    print('\n============================= FEATURE-SELECTION =============================\n')
    print('> Manual Feature-Pre-Selection:')
    for feature in feature_preselection:
        print(' ',feature, end=',')
    print('\n\n> Imported Dataset after Feature-Pre-Selection:\t',X.shape)


    # b) Automatic Feature-Selection:

    # Univariate automatic feature selection:

    # Applying SelectKBest class to extract top best features:
    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(X, y)

    data_scores = pd.DataFrame(fit.scores_)
    data_columns = pd.DataFrame(X.columns)
    # Concat two dataframes for better visualization:
    feature_scores = pd.concat([data_columns, data_scores],axis=1)
    feature_scores.columns = ['Feature','Score']
    feature_scores.sort_values(by=['Score'])

    # Removing features with low variance:
    #remover = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #X = remover.fit_transform(X)
    #print(X.iloc[0]) # =header


    # V. Data division into Training and Test datasets & Normalization:
    ratio = 0.3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)

    print('\n============================= DATASET DIVISION ==============================\n')
    print('> split ratio:\t',ratio,'/',1.0-ratio)
    print('> Testset:\t',X_test.shape)
    print('> Trainingset:\t',X_train.shape)
    print('> Full useable / imported Dataset:',data_filtered.shape,'/',data.shape,'\n')


    # Normalization (L1 & L2):
    # NOTE:  Change 'normtype' value to 'l1' / 'l2' to change normalization type:
    normtype = 'l2'#'l1'
    print('> Normalizing datasets with ', normtype,'\n')
    X_test= normalize(X_test, norm=normtype)
    X_train = normalize(X_train, norm=normtype)



    # VI. Prediction & Classification:

    # model_selection is used for manually enabeling the individual models.
    # NOTE:  setting boolean value, eanbles/disables model
    model_selection = {
        'ExtraTrees': ( True, ExtraTreesClassifier() ),
        'RandomForest': ( True, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) ),
        'AdaBoost': ( True, AdaBoostClassifier() ),
        'DecisionTree': ( True, DecisionTreeClassifier(max_depth=5) ),
        'NearestNeighbors': (True, KNeighborsClassifier(n_neighbors=5) ), # (n_neighbors=4) ),
        'LinearSVM': ( True, SVC(kernel="linear", C=0.025) ),  # (C=0.01, penalty="l1", dual=False) ),
        'RBF_SVM': (True, SVC(gamma=2, C=1) ), # (gamma='auto') ),
        'Nu_SVM': (True, NuSVC() ),
        'GaussianProcess': (True, GaussianProcessClassifier() ), #(1.0 * RBF(1.0)) ),
        'NeuralNet': (True, MLPClassifier(alpha=1, max_iter=1000) ),
        'LogisticRegression': (True, LogisticRegression() ),
        'QDA': (True, QuadraticDiscriminantAnalysis() ),
        'LDA': (True, LinearDiscriminantAnalysis() ),
        'NaiveBayes': (True,  GaussianNB() ),
        'GradientBoosting': (True, GradientBoostingClassifier() ),
        'RadiusNeighborsClassifier': (True, RadiusNeighborsClassifier() ),
        'SGDClassifier': (True, SGDClassifier() ),
        'RidgeClassifierCV': (True, RidgeClassifierCV() ),
        'RidgeClassifier': (True, RidgeClassifier() ),
        'PassiveAggressiveClassifier': (True, PassiveAggressiveClassifier() ),
        'BaggingClassifier': (True, BaggingClassifier() ),
        'BernoulliNB': (True, BernoulliNB() ),
        'CalibratedClassifierCV': (True, CalibratedClassifierCV() ),
        'LabelPropagation': (True, LabelPropagation() ),
        'LabelSpreading': (True, LabelSpreading() ),
        'LinearSVC': (True, LinearSVC() ),
        'LogisticRegressionCV': (True, LogisticRegressionCV() ),
        'MultinomialNB': (True, MultinomialNB() ),
        'NearestCentroid': (True, NearestCentroid() ),
        'Perceptron': (True, Perceptron() )
        #'OneClassSVM': (True, OneClassSVM() ),
        #'ClassifierChain': (True, ClassifierChain() ),
        #'MultiOutputClassifier': (True, MultiOutputClassifier() ),
        #'OutputCodeClassifier': (True, OutputCodeClassifier() ),
        #'OneVsOneClassifier': (True, OneVsOneClassifier() ),
        #'OneVsRestClassifier': (True, OneVsRestClassifier() ),
    }

    estimate = Estimate(model_selection, X_test, y_test, X_train, y_train, feature_preselection)
    results_df, importances = estimate.results


    print('\n============================== FEATURE RANKING ==============================\n')
    importance_ranking = rank_feature_importance(importances)

    print('> Table containing importance of every feature with different classifiers:\n')
    print(importances.to_string(),'\n')


    print('> Features with highest importance with different classifiers:')
    print(importance_ranking)


    print('\n> Univariate automatic feature selection:\nApplying SelectKBest class to extract top best features:')
    print(feature_scores)
    #print(feature_scores.round({'Score': 3}))
    #print(feature_scores.nlargest(20,'Score', ))  #print n best features


    print('\n========================== PREDICTION MODEL RANKING ==========================\n')
    print(results_df)

    print('\n=============================================================================')



if __name__ == '__main__':
    main()
