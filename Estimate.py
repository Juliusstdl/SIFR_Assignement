#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Estimate:
# Fitting data to different models. Prediction, Classification and Evaluation of the different models.

import operator
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel

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

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import model_selection


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

        results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'MSE'])#, 'Score'])
        importances_df = pd.DataFrame(columns=feature_selection)

        for model_name, tupel in model_selection.items():
            active, model = tupel
            if active:

                model.fit(X_train, y_train)

                # a) Get the predictions of the model from the data it has not seen (testing):
                y_pred_test = model.predict(X_test)


                # b) Evaluation of the model:
                accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_test)#, normalize=True, sample_weight=None)

                precision = precision_score(y_true=y_test, y_pred=y_pred_test)#, labels=None, pos_label=1, average=’binary’, sample_weight=None)

                recall = recall_score(y_true=y_test, y_pred=y_pred_test)#, labels=None, pos_label=1, average=’binary’, sample_weight=None)

                f1 = f1_score(y_true=y_test, y_pred=y_pred_test)#, labels=None, pos_label=1, average=’binary’, sample_weight=None)

                mse = mean_squared_error(y_true=y_test.astype(np.float64), y_pred=y_pred_test.astype(np.float64))#y_pred=y_pred_test.astype(np.float64), y_true=y_test.astype(np.float64))

                #crossval = cross_val_score()

                #score = model.score(X_test, y_test)
                #score = model.score(y_true=y_test, y_pred=y_pred_test)


                results_df = results_df.append( { 'Model': model_name, 'Accuracy': accuracy, 'F1-Score': f1, 'Precision': precision, 'Recall': recall, 'MSE': mse}, ignore_index=True )#, 'Score': score

                '''
                # All the metrics compare in some way how close are the predicted vs. the actual values:
                error_metric = mean_squared_error(y_pred=y_pred_test.astype(np.float64), y_true=y_test.astype(np.float64))
                #print('> The Mean Square Error of the',model_name,'-model is: ',error_metric,'\n')

                # Cross Validation Classification Accuracy:
                #seed = 7
                #kfold = model_selection.KFold(n_splits=10, random_state=seed)
                #scoring = 'accuracy'
                #results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
                #print("Accuracy: %.3f (%.3f)") % (results.mean(), results.std())
                '''

                # c) Get feature importances:
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


        results_df = results_df.sort_values('Accuracy', ascending=False) #(by=['Accuracy'])

        return (results_df, importances_df)
