#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# @author:   Julius Steidl
# date:     25.07.2019
# version:  1.8
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
#from sklearn.ensemble.voting_classifier import VotingClassifier
#from sklearn.mixture import DPGMM
#from sklearn.mixture import GMM
#from sklearn.mixture import GaussianMixture
#from sklearn.mixture import VBGMM

# Muting the warning-messages:
pd.options.mode.chained_assignment = None  # default='warn'


# NOTE:  Set 'directory'-string to name of the folder containing the .csv input files:
directory = 'Kickstarter_2019-07-18T03_20_05_009Z'  # date: 2019-07-18


# Normalization (L1 & L2):
# NOTE:  Change 'normtype' value to 'l1' / 'l2' to change normalization type:
normtype = 'l2'#'l1'


# model_selection is used for manually enabling the individual models.
# NOTE:  Setting boolean value, eanbles/disables model.
model_selection = {
	'ExtraTrees': ( True, ExtraTreesClassifier() ),
	'RandomForest': ( True, RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) ),
	'AdaBoost': ( True, AdaBoostClassifier() ),
	'DecisionTree': ( True, DecisionTreeClassifier(max_depth=5) ),
	'NearestNeighbors': (True, KNeighborsClassifier(n_neighbors=5) ), # (n_neighbors=4) ),
	'LinearSVM': ( True, SVC(kernel="linear", C=0.025) ),  # (C=0.01, penalty="l1", dual=False) ),
	'RBF_SVM': (True, SVC(gamma='auto') ),#gamma=2, C=1) ), #
	'Nu_SVM': (True, NuSVC(gamma='auto') ),
	'GaussianProcess': (False, GaussianProcessClassifier() ), #(1.0 * RBF(1.0)) ),
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
	'Perceptron': (True, Perceptron() ),
	#'OneClassSVM': (True, OneClassSVM() ),
	#'ClassifierChain': (True, ClassifierChain() ),
	#'MultiOutputClassifier': (True, MultiOutputClassifier() ),
	#'OutputCodeClassifier': (True, OutputCodeClassifier() ),
	#'OneVsOneClassifier': (True, OneVsOneClassifier() ),
	#'OneVsRestClassifier': (True, OneVsRestClassifier() ),
}


# feature_set is used for manually enabling the individual features.
# NOTE:  setting boolean value, eanbles/disables feature.
feature_set = {
    'backers_count': True,
    'converted_pledged_amount': True,
    'goal': True,
    'country': True,
    'staff_pick': True,
    'spotlight': True,
    'launched_at': True,
    'deadline': True,
    'cat_id': True,
    'cat_name': True,
    'subcat_name': True,
    'pos': True,
    'parent_id': True,
    'person_id': True,
    'person_name': True,
    'location_id': True,
    'location_name': True,
    'location_state': True,
    'location_type': True,
    'duration_days': True,
    'goal_exceeded': True,
    'divergence': False # feature contains negative value!
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
    				print('>>> Please enter a float value between 0.0 and 1.0 to determine the slice of the imported dataset beeing used >>>\n')
    		# Skipping new dataset import and statistics generation:
    		if arg in ['-skipall', 'skipall', '-skip', 'skip']:
    			skipimport = True; skipstats = True
    		elif arg in ['-skipimport', 'skipimport']:
    			skipimport = True
    		elif arg in ['-skipstats', 'skipstats', '-skipstatistics', 'skipstatistics']:
    			skipstats = True

    return (skipimport, skipstats, slice_value)


def main():
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
    data_filtered = data.loc[data['state'].isin(['successful','failed'])]

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
            print('  -',feature,'\t\tused')
        else:
            print('  -',feature,'\t\tnot used')

    # Dividing dataset into X (=features), and y (=labels):
    X = data_encoded[feature_subset]
    y = data_encoded['state']

    # b) Automatic Feature-Selection:

    # Univariate automatic feature selection:

    # Applying SelectKBest class to extract top best features:
    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(X, y)

    data_scores = pd.DataFrame(fit.scores_)
    data_columns = pd.DataFrame(X.columns)
    # Concat two dataframes for better visualization:
    feature_scores_df = pd.concat([data_columns, data_scores], axis=1)
    feature_scores_df.columns = ['Feature', 'Score']
    feature_scores_df = feature_scores_df.round(5)
    feature_scores_df = feature_scores_df.sort_values(by=['Score'], ascending=False)

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
    ratio = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio)

    print('\n=============================== DATASET SIZE ================================\n')
    print('> Full useable / imported dataset: ',data_filtered.shape,' / ',data.shape)
    if slice_value < 1.0:
        print('> Used dataset slice: ',X.shape,' = ',slice_value*100.0,'%   (as defined by user argument)')
    print('> Testset:\t',X_test.shape)
    print('> Trainingset:\t',X_train.shape)
    print('> Split ratio:  ',ratio,'/',1.0-ratio)


    # Normalization (L1 & L2):
    print('\n> Normalizing datasets with ', normtype)
    X_test= normalize(X_test, norm=normtype)
    X_train = normalize(X_train, norm=normtype)



    # VI. Estimation: Prediction & Classification:

    print('\n================================= ESTIMATING ================================\n')

    estimate = Estimate(model_selection, X_test, y_test, X_train, y_train, feature_subset)
    results_df, importances_df = estimate.results


    print('\n============================== FEATURE RANKING ==============================\n')

    print('> Table containing importance of every feature with different classifiers:\n')
    print(importances_df.to_string(),'\n')
    importances_figure = importances_df.plot.barh(stacked=True)
    plt.savefig('feature_importances.png')
    plt.show()


    print('> Features with highest importance with different classifiers:')
    importances_sum = importances_df.sum(axis = 0, skipna = True)
    importances_sum_df = importances_sum.to_frame()
    importances_sum_df.columns = ['Importance']
    importances_sum_df = importances_sum_df.sort_values(by=['Importance'], ascending=False)

    print(importances_sum_df.to_string(),'\n')


    print('\n> Univariate automatic feature selection:  Applying SelectKBest class to extract top best features:')
    print(feature_scores_df)
    #print(feature_scores_df.round({'Score': 3}))
    #print(feature_scores_df.nlargest(20,'Score', ))  #print n best features


    print('\n========================== PREDICTION MODEL RANKING ==========================\n')
    print(results_df.to_string())
    results_df.plot.barh(x='Model')
    plt.show()
    plt.savefig('pred_model_rank.png')

    print('\n=============================================================================')



if __name__ == '__main__':
    main()
