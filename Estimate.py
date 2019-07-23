#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Estimate:
# Fitting data to different models. Prediction, Classification and Evaluation of the different models.

import operator
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn import model_selection

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


class Estimate:

    def __init__(self, model_selection, X_test, y_test, X_train, y_train, feature_preselection):
        self.model_selection = model_selection
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.feature_preselection = feature_preselection
        self.results = self.estimate(model_selection, X_test, y_test, X_train, y_train, feature_preselection)


    def estimate(self, model_selection, X_test, y_test, X_train, y_train, feature_selection):

        scores = {}; errors = {}
        importances_df = pd.DataFrame(columns=feature_selection)

        for model_name, tupel in model_selection.items():
            active, model = tupel
            if active:
                model.fit(X_train, y_train)

                # a) Evaluation of the model:
                '''MAE is the average of the absolute difference between the predicted values and observed value.
                The MAE is a linear score which means that all the individual differences are weighted equally in the average.
                For example, the difference between 10 and 0 will be twice the difference between 5 and 0.'''
                try:
                    # Get the predictions of the model fro the data it has not seen (testing):
                    y_pred_test = model.predict(X_test)

                    # All the metrics compare in some way how close are the predicted vs. the actual values:
                    error_metric = mean_squared_error(y_pred=y_pred_test.astype(np.float64), y_true=y_test.astype(np.float64))

                    errors[model_name] = error_metric

                    #print('> The Mean Square Error of the',model_name,'-model is: ',error_metric,'\n')

                except:
                    print('Prediction error with using',model_name,'as model.\n')
                    continue


                # b) Get scores of Classification:
                score = model.score(X_test, y_test)

                scores[model_name] = score

                # Cross Validation Classification Accuracy:
                #seed = 7
                #kfold = model_selection.KFold(n_splits=10, random_state=seed)
                #scoring = 'accuracy'
                #results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                #print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())


                # c) get feature importances:
                try:
                    feature_importances = model.feature_importances_
                    importances = {}
                    indices = np.argsort(feature_importances)

                    for i, value in zip(indices, feature_importances):
                        importances[feature_selection[i]] = value

                    importances_df.loc[model_name] = importances

                except:
                    #print(' -',model_name,'does not output any feature importances.')
                    continue

            # Feature selection using SelectFromModel:
            #cls_LSV = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
            #model = SelectFromModel(cls_LSV, prefit=True)
            #X_train_new = model.transform(X_train)


        # Sorting error & score values and saving them in Dataframe:
        errors = sorted(errors.items(), key=operator.itemgetter(1))
        errors_df = pd.DataFrame.from_dict(errors)
        errors_df.columns = ['Model','SquareError']

        scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        scores_df = pd.DataFrame.from_dict(scores)
        scores_df.columns = ['Model','Score']

        results_df = pd.merge(errors_df, scores_df, on='Model')
        results_df.sort_values(by=['SquareError'])

        return (results_df, importances_df)
