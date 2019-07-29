#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# @author:   Julius Steidl
# date:     29.07.2019
# version:  2.0
# NOTE:     folder with .csv files is required:  ./Kickstarter_2019-07-18T03_20_05_009Z/
#           source:  https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2019-07-18T03_20_05_009Z.zip


from DatasetImport import DatasetImport
from Statistics import Statistics
from Preprocessing import Preprocessing
from Estimate import Estimate

import sys
import os
import pandas as pd
import numpy as np
import ast
import re
import operator
from collections import defaultdict
import matplotlib.pyplot as plt

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


# Muting the warning-messages:
pd.options.mode.chained_assignment = None  # default='warn'


# NOTE:  Set 'directory'-string to name of the folder containing the .csv input files:
directory = 'Kickstarter_2019-07-18T03_20_05_009Z'  # date: 2019-07-18


# Filtering out entries (=campaigns) with 'canceled' or 'live' state values (as label).
# => Just using 'successful' and 'failed' campaigns for estimation:
state_filter = ['successful','failed'] #,'canceled', 'live']


# NOTE:  Adjust Trainingset / Testset division ratio:
divratio = 0.3


# Normalization (L1 & L2):
# NOTE:  Change 'normtype' value to 'l1' / 'l2' to change normalization type:
normtype = 'l2'#'l1'


# model_selection is used for manually enabling the individual models.
# NOTE:  Setting boolean value, eanbles/disables model.
model_selection = {
    'ExtraTrees': ( True, ExtraTreesClassifier(n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None) ),
    'RandomForest': ( True, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) ),
    'AdaBoost': ( True, AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None) ),
    'DecisionTree': ( True, DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False) ),
    'GradientBoosting': (True, GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001) ),
    'BernoulliNB': (True, BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None) ),
    'BaggingClassifier': (True, BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0) ),
    'NearestNeighbors': (True, KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None) ), # (n_neighbors=4) ),
    'LogisticRegressionCV': (True, LogisticRegressionCV(Cs=10, fit_intercept=True, cv='warn', dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0, multi_class='warn', random_state=None, l1_ratios=None) ),
    'LDA': (True, LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001) ),
    'LogisticRegression': (True, LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None) ),
    'CalibratedClassifierCV': (True, CalibratedClassifierCV(base_estimator=None, method='sigmoid', cv='warn') ),
    'LinearSVC': (True, LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000) ),
    'LinearSVM': ( True, SVC(kernel='linear', C=0.025) ),  # (C=0.01, penalty='l1', dual=False) ),
    'RBF_SVM': (True, SVC(gamma='auto') ),#gamma=2, C=1) ), #
    'Nu_SVM': (True, NuSVC(gamma='auto') ),
    'GaussianProcess': (False, GaussianProcessClassifier() ), #(1.0 * RBF(1.0)) ),
    'NeuralNet': (True, MLPClassifier(alpha=1, max_iter=1000) ),
    'QDA': (True, QuadraticDiscriminantAnalysis() ),
    'NaiveBayes': (True,  GaussianNB() ),
    'RadiusNeighborsClassifier': (True, RadiusNeighborsClassifier() ),
    'SGDClassifier': (True, SGDClassifier() ),
    'RidgeClassifierCV': (True, RidgeClassifierCV() ),
    'RidgeClassifier': (True, RidgeClassifier() ),
    'PassiveAggressiveClassifier': (True, PassiveAggressiveClassifier() ),
    'LabelPropagation': (True, LabelPropagation() ),
    'LabelSpreading': (False, LabelSpreading() ),
    'MultinomialNB': (True, MultinomialNB() ),
    'NearestCentroid': (True, NearestCentroid() ),
    'Perceptron': (True, Perceptron() ),
}


# feature_set is used for manually enabling the individual features.
# NOTE:  setting boolean value, eanbles/disables feature.
feature_set = {
    'backers_count': True,
    'converted_pledged_amount': True,
    'goal': True,
    'country': True,
    'staff_pick': False,
    'spotlight': True,
    'launched_at': False,
    'deadline': False,
    'cat_id': True,
    'cat_name': False,
    'subcat_name': True,
    'pos': True,
    'parent_id': True,
    'person_id': True,
    'person_name': False,
    'location_id': True,
    'location_name': False,
    'location_state': True,
    'location_type': True,
    'duration_days': True, # extracted feature
    'duration_median': False, # extracted feature
    'year': True, # extracted feature
    'month': True, # extracted feature
    'goal_exceeded': True, # extracted feature
    'divergence': True # extracted feature, negative value
}
# labels = ['state', 'id', 'name']



def handle_arguments(argv):
    skipimport = False; skipstats = False; slice_value = 1.0

    if len(argv[1:]) > 1:
        for arg in argv:
            if re.match('^\d\.\d+?$', arg) is not None:
                if float(arg) < 1.0 and float(arg) > 0.0:
                    slice_value = float(arg)
                    print('>>> Entered slice value:',slice_value)
                else:
                    print('>>> Input Error: Please enter a float value between 0.0 and 1.0 to determine the slice of the imported dataset beeing used >>>\n')
            # Skipping new dataset import and statistics generation:
            if arg in ['-skipall', 'skipall', '-skip', 'skip']:
                skipimport = True; skipstats = True
            elif arg in ['-skipimport', 'skipimport']:
                skipimport = True
            elif arg in ['-skipstats', 'skipstats', '-skipstatistics', 'skipstatistics']:
                skipstats = True

    return (skipimport, skipstats, slice_value)


def univariate_selection(X, y):

    # Applying SelectKBest class to extract top best features:
    bestfeatures = SelectKBest(score_func=chi2, k='all')
    fit = bestfeatures.fit(X, y)

    data_scores = pd.DataFrame(fit.scores_)
    data_columns = pd.DataFrame(X.columns)
    # Concat two dataframes for better visualization:
    feature_scores_df = pd.concat([data_columns, data_scores], axis=1)
    feature_scores_df.columns = ['Feature', 'Score']
    feature_scores_df = feature_scores_df.round(5)
    feature_scores_df = feature_scores_df.sort_values(by=['Score'], ascending=False)

    return feature_scores_df



def main():
    print('>>> Starting campaign success predictor, v2.0, 29.07.2019, by Julius Steidl >>>')

    # Handling user arguments, if existing:
    skipimport, skipstats, slice_value = handle_arguments(sys.argv)


    # I. Importing Dataset & Data Cleaning:

    # saving / loading dataframe as .pickle file  (-> not necessary to parse .csv-files every time):
    if skipimport and os.path.exists(directory+'.pickle') and os.path.isfile(directory+'.pickle'):
        data = pd.read_pickle('./'+directory+'.pickle')
    else:
        datasetImport = DatasetImport(directory)
        data = datasetImport.all_data
        data.to_pickle(directory+'.pickle')



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
    data_filtered = data.loc[data['state'].isin(state_filter)]

    # Encoding string-value columns to logical datatypes (binary & ordinal) for estimation:
    preprocessing = Preprocessing(data_filtered)
    data_encoded = preprocessing.data_encoded



    # IV. Feature-Selection:

    # a) Manual Feature-Pre-Selection:
    print('\n============================= FEATURE-SELECTION =============================\n')
    print('> Manual Feature-Pre-Selection:')

    feature_subset = []
    for feature, active in feature_set.items():
        if active:
            feature_subset.append(feature)
            print('  - active   ',feature)
        else:
            print('  - inactive ',feature,)

    # Dividing dataset into X (=features), and y (=labels):
    X = data_encoded[feature_subset]
    y = data_encoded['state']


    # b) Automatic Feature-Selection:

    # Univariate automatic feature selection:
    feature_scores_df = univariate_selection(X, y)


    # Removing features with low variance:
    #remover = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #X = remover.fit_transform(X)
    #print(X.iloc[0]) # =header



    # V. Data division into Training and Test datasets & Normalization:

    # Using the user argument value to set the set the slice of the imported dataset beeing used:
    rows, cols = data_filtered.shape
    if slice_value < 1.0:
        slice = int(round(float(rows) * slice_value))
        X = X.head(slice)
        y = y.head(slice)


    # Dividing dataset into Training and Test datasets according to predifined ratio:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=divratio)

    print('\n=============================== DATASET SIZE ================================\n')
    print('> Full useable / imported dataset: ',data_filtered.shape,' / ',data.shape)
    if slice_value < 1.0:
        print('> Used dataset slice: ',X.shape,' = ',slice_value*100.0,'%   (as defined by user argument)')
    print('> Testset:\t',X_test.shape)
    print('> Trainingset:\t',X_train.shape)
    print('> Split ratio:  ',divratio,'/',1.0-divratio)


    # Normalization (L1 & L2):
    print('\n> Normalizing datasets with ', normtype)
    X_test= normalize(X_test, norm=normtype)
    X_train = normalize(X_train, norm=normtype)



    # VI. Estimation: Prediction & Classification:

    #print('\n================================= ESTIMATING ================================\n')

    estimate = Estimate(model_selection, X_test, y_test, X_train, y_train, feature_subset)
    results_df, importances_df = estimate.results


    print('\n============================== FEATURE RANKING ==============================\n')
    print('> FEATURE IMPORTANCE:  Models supporting feature_importances_:\n  ',list(importances_df.index),'\n')

    #print('> Table containing importance of every feature (only from models supporting feature_importances_):')
    #print(importances_df.to_string(),'\n')
    importances_figure = importances_df.plot.barh(stacked=True)
    plt.savefig('feature_importances.png')
    plt.show()

    print('> Features with highest importance (only from models supporting feature_importances_):')
    importances_sum = importances_df.sum(axis = 0, skipna = True)
    importances_sum_df = importances_sum.to_frame()
    importances_sum_df.columns = ['Importance']
    importances_sum_df = importances_sum_df.sort_values(by=['Importance'], ascending=False)

    print(importances_sum_df.to_string(),'\n')


    print('\n> Univariate automatic feature selection:  Applying SelectKBest class to extract best features:')
    print(feature_scores_df)
    feature_scores_df.plot.bar(x='Feature', logy=True)
    plt.savefig('univariate_selection.png')
    plt.show()


    print('\n========================== PREDICTION MODEL RANKING ==========================\n')
    print(results_df.to_string())
    results_df.plot.barh(x='Model')
    plt.show()
    plt.savefig('pred_model_rank.png')

    print('\n=============================================================================')



if __name__ == '__main__':
    main()
