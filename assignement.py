#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# author:   Julius Steidl
# date:     18.07.2019
# version:  1.4
# note:     directory with .csv files:  ./Kickstarter_2019-04-18T03_20_02_220Z/

import os
import pandas as pd
import numpy as np
import ast
import re
import operator
from datetime import datetime
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectFromModel
'''
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
'''
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


pd.options.mode.chained_assignment = None  # default='warn'


def import_datasets(directory):
    print('> Importing .csv-files from folder: ',directory)
    filecount = 0

    # Iterating over all .csv files in subdirectory:
    for filename in os.listdir(directory):

        if filename.endswith(".csv"):
            print(' '+str(filecount)+': '+filename)
            filecount += 1

            with open(directory+'/'+filename, 'r', encoding="utf8", newline='') as csvfile:

                raw_data = pd.read_csv('Kickstarter_2019-04-18T03_20_02_220Z/'+filename, header=None)

                # selecting dataframe columns for import:
                data = raw_data[[31,16,22,0,3,15,4,30,29,20,11]]

                # using first line as index:
                headers = data.iloc[0]
                data = pd.DataFrame(data.values[1:], columns=headers)


                # Parsing 'category'-column string as individual columns/features:
                cat_id = []; cat_name = []; subcat_name = []; pos = []; parent_id = []

                for string in [str(x) for x in raw_data[2][1:]]:

                    cat_mask = re.compile('{"id":(\d+?),.+"slug":"(.+?)(\/(.+?))?","position":(\d+?)(,"parent_id":(\d+?))?,')
                    extract = re.search(cat_mask, string)

                    if extract and extract.group(1) is not None:
                        cat_id.append(int(extract.group(1)))
                    else:
                        cat_id.append(0)

                    if extract and extract.group(2) is not None:
                        cat_name.append(extract.group(2))
                    else:
                        cat_name.append('null')

                    if extract and extract.group(4) is not None:
                        subcat_name.append(extract.group(4))
                    else:
                        subcat_name.append('null')

                    if extract and extract.group(5) is not None:
                        pos.append(int(extract.group(5)))
                    else:
                        pos.append(0)

                    if extract and extract.group(7) is not None:
                        parent_id.append(int(extract.group(7)))
                    else:
                        parent_id.append(0)


                # Parsing 'creator'-column string as individual columns/features:
                person_id = []; person_name = [];

                for string in [str(x) for x in raw_data[6][1:]]:
                    string = str(string)

                    person_mask = re.compile('{"id":(\d+?),"name":"(.+?)"')
                    extract = re.search(person_mask, string)

                    if extract and extract.group(1) is not None:
                        person_id.append(int(extract.group(1)))
                    else:
                        person_id.append(0)

                    if extract and extract.group(2) is not None:
                        person_name.append(extract.group(2))
                    else:
                        person_name.append('null')


                # Parsing 'location'-column string as individual columns/features:
                location_id = []; location_name = []; location_state = []; location_type = []

                for string in [str(x) for x in raw_data[21][1:]]:

                    location_mask = re.compile('{"id":(\d+?),.+"short_name":"(.+?)".+"state":"(.+?)","type":"(.+?)"')
                    extract = re.search(location_mask, string)

                    if extract and extract.group(1) is not None:
                        location_id.append(int(extract.group(1)))
                    else:
                        location_id.append(0)

                    if extract and extract.group(2) is not None:
                        location_name.append(extract.group(2))
                    else:
                        location_name.append('null')

                    if extract and extract.group(3) is not None:
                        location_state.append(extract.group(3))
                    else:
                        location_state.append('null')

                    if extract and extract.group(4) is not None:
                        location_type.append(extract.group(4))
                    else:
                        location_type.append('null')


                # Calculating campaign duration:
                duration = []; divergence = []; goal_exceeded = []

                for index, row in data.iterrows():
                    launched_at = row['launched_at']
                    deadline = row['deadline']
                    difference = int(round((float(deadline)-float(launched_at))/(60*60*24)))
                    duration.append(difference)
                    #print('launched at:',datetime.utcfromtimestamp(launched_at).strftime('%d-%m-%Y'), '- deadline:',datetime.utcfromtimestamp(deadline).strftime('%d-%m-%Y'), '- difference in days:',difference)

                    # Calculating divergence between pledged money and goal:
                    # divergence measeures the divergence value of pledged from goal in USD and can be positive or negative
                    # goal_exceeded is True if pledged money exceeds the goal

                    div = float(row['converted_pledged_amount']) - float(row['goal'])
                    divergence.append(div)

                    if div >= 0:
                        goal_exceeded.append(True)
                    else:
                        goal_exceeded.append(False)


            data = data.assign( cat_id = cat_id, cat_name = cat_name, subcat_name = subcat_name, pos = pos, parent_id = parent_id, person_id = person_id, person_name = person_name, location_id = location_id, location_name = location_name, location_state = location_state, location_type = location_type, duration = duration, divergence = divergence, goal_exceeded = goal_exceeded)
            # Coverting string-values in dataframe to numeric- & boolean-values:
            data['id'] = pd.to_numeric(data['id'])
            data['backers_count'] = pd.to_numeric(data['backers_count'])
            data['converted_pledged_amount'] = pd.to_numeric(data['converted_pledged_amount'])
            data['goal'] = pd.to_numeric(data['goal'])

            bool_dict = {'true': True,'false': False}
            data['staff_pick'] = data['staff_pick'].map(bool_dict)
            data['spotlight'] = data['spotlight'].map(bool_dict)

            # Rearranging column sequence:
            columns = ['state', 'id', 'name', 'backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence', 'divergence', 'goal_exceeded']

            data = data[columns]

        else:
            continue

    return data


def generate_statistics(data):

    campaign_count, c = data.shape
    successful = data[data['state'].str.contains('successful')]
    unsuccessful = data[data['state'].str.contains('failed','canceled')]

    success_count, c = successful.shape
    unsuccess_count, c = unsuccessful.shape
    backers_success = successful['backers_count'].sum()
    backers_unsuccess = unsuccessful['backers_count'].sum()
    pledged_success = successful['converted_pledged_amount'].sum()
    pledged_unsuccess = unsuccessful['converted_pledged_amount'].sum()
    goal_success = successful['goal'].sum()
    goal_unsuccess = unsuccessful['goal'].sum()
    pick_success = successful['staff_pick'].sum()
    pick_unsuccess = unsuccessful['staff_pick'].sum()
    spot_success = successful['spotlight'].sum()
    spot_unsuccess = unsuccessful['staff_pick'].sum()
    duration_success = successful['duration'].sum()
    duration_unsuccess = unsuccessful['duration'].sum()

    print('\n========================= EXPLORATORY DATA ANALYSIS =========================')

    print('> Total number of campaigns: ',campaign_count)
    print('\t- backers avg.:\t',round(float((backers_success+backers_unsuccess)/float(campaign_count))))
    print('\t- pledged avg.:\t',round(float((pledged_success+pledged_unsuccess)/float(campaign_count))),'$')
    print('\t- goal avg.:\t',round(float((goal_success+goal_unsuccess)/float(campaign_count))),'$')
    print('\t- staff pick:\t',round(100 * float(pick_success+pick_unsuccess)/float(campaign_count)),'%')
    print('\t- spotlight:\t',round(100 * float(spot_success+spot_unsuccess)/float(campaign_count)),' %')
    print('\t- duration avg.:',round(float((duration_success+duration_unsuccess)/float(campaign_count))),'days')

    print('\n> successful:  ',round(100 * float(success_count)/float(campaign_count)),'%')
    print('\t- backers avg.:\t',round(float(backers_success/float(success_count))))
    print('\t- pledged avg.:\t',round(float(pledged_success/float(success_count))),'$')
    print('\t- goal avg.:\t',round(float(goal_success/float(success_count))),'$')
    print('\t- staff pick:\t',round(100 * float(pick_success)/float(success_count)),'%')
    print('\t- spotlight:\t',round(100 * float(spot_success)/float(success_count)),'%')
    print('\t- duration avg.:',round(float((duration_success)/float(success_count))),'days')

    print('\n> unsuccessful:  ',round(100 * float(unsuccess_count)/float(campaign_count)),'%')
    print('\t- backers avg.:\t',round(float(backers_unsuccess/float(unsuccess_count))))
    print('\t- pledged avg.:\t',round(float(pledged_unsuccess/float(unsuccess_count))),'$')
    print('\t- goal avg.:\t',round(float(goal_unsuccess/float(unsuccess_count))),'$')
    print('\t- staff pick:\t',round(100 * float(pick_unsuccess)/float(unsuccess_count)),'%')
    print('\t- spotlight:\t',round(100 * float(spot_unsuccess)/float(unsuccess_count)),'%')
    print('\t- duration avg.:',round(float((duration_unsuccess)/float(unsuccess_count))),'days')

    print('\n============================ TOP (UN)SUCCESSFUL =============================')

    top = 5

    print('\n> Categories with most successful campaigns:')
    print(successful.cat_name.value_counts()[:top])
    print('\n> Categories with most unsuccessful campaigns:')
    print(unsuccessful.cat_name.value_counts()[:top])

    print('\n> Subcategories with most successful campaigns:')
    print(successful.subcat_name.value_counts()[:top])
    print('\n> Subcategories with most unsuccessful campaigns:')
    print(unsuccessful.subcat_name.value_counts()[:top])

    print('\n> Persons with most successful campaigns:')
    print(successful.person_name.value_counts()[:top])
    print('\n> Persons with most unsuccessful campaigns:')
    print(unsuccessful.person_name.value_counts()[:top])

    print('\n> Countries with most successful campaigns:')
    print(successful.country.value_counts()[:top])
    print('\n> Countries with most unsuccessful campaigns:')
    print(unsuccessful.country.value_counts()[:top])

    print('\n> Locations with most successful campaigns:')
    print(successful.location_name.value_counts()[:top])
    print('\n> Locations with most unsuccessful campaigns:')
    print(unsuccessful.location_name.value_counts()[:top])

    print('\n> States with most successful campaigns:')
    print(successful.location_state.value_counts()[:top])
    print('\n> States with most unsuccessful campaigns:')
    print(unsuccessful.location_state.value_counts()[:top])


def feature_encoding(data):
    labelencoder = LabelEncoder()

    # data = ['state', 'id', 'name', 'backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence', 'goal_exceeded']

    # Converting string values from 'state'-column into binary integers (0 / 1):
    bool_dict = {'successful': True,'failed': False}
    data['state'] = data['state'].map(bool_dict)

    data['country'] = labelencoder.fit_transform(data['country'])
    data['cat_name'] = labelencoder.fit_transform(data['cat_name'])
    data['subcat_name'] = labelencoder.fit_transform(data['subcat_name'])
    data['person_name'] = labelencoder.fit_transform(data['person_name'])
    data['location_name'] = labelencoder.fit_transform(data['location_name'])
    data['location_state'] = labelencoder.fit_transform(data['location_state'])
    data['location_type'] = labelencoder.fit_transform(data['location_type'])
    #data['duration'] = labelencoder.fit_transform(data['duration'])

    #categories = labelencoder.categories_


    # Encoding (Ordinal Encoder & One Hot Encoder):
    #ordinalencoder = preprocessing.OrdinalEncoder()
    #onehotencoder = preprocessing.OneHotEncoder()
    #encoder.fit(data)
    #encoder.transform([['foo','bar']])
    #encoder.transform([['foo','bar'],['baz','quux']]).toarray()

    #encoder = preprocessing.OneHotEncoder(categories=[categories, locations])

    # One Hot Encoding features:
    #data_dummies = pd.get_dummies(data)
    #print(list(data.columns))
    #print(list(data_dummies.columns))

    return data


def classify(features_test, labels_test, features_train, labels_train, classifiers, feature_selection):

    scores = {}
    importances_df = pd.DataFrame(columns=feature_selection)

    for cls_name, tupel in classifiers.items():
        active, classifier = tupel
        if active:
            classifier.fit(features_train, labels_train)

            score = classifier.score(features_test, labels_test)
            scores[cls_name] = score

            # get feature importances:
            try:
                feature_importances = classifier.feature_importances_
                importances = {}
                indices = np.argsort(feature_importances)

                for i, value in zip(indices, feature_importances):
                    importances[feature_selection[i]] = value

                importances_df.loc[cls_name] = importances
            except:
                continue

        # Feature selection using SelectFromModel:
        #cls_LSV = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data_features, data_labels)
        #model = SelectFromModel(cls_LSV, prefit=True)
        #features_train_new = model.transform(features_train)

    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    scores_df = pd.DataFrame.from_dict(scores)
    scores_df.columns = ['Classifier','Score']

    return (scores_df, importances_df)


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

    directory = 'Kickstarter_2019-04-18T03_20_02_220Z'  # date: 2019-05-16

    # saving / loading dataframe as .pickle file  (-> not necessary to parse .csv-files every time):
    # NOTE:  uncomment next two lines for skipping import of new dataset from .csv-files:
    data_imported = import_datasets(directory)
    data_imported.to_pickle(directory+'.pickle')

    data = pd.read_pickle('./'+directory+'.pickle')



    # II. Exploratory data analysis:

    # Generating some statistics for getting insight into dataset & for supporting manual feature-selection decisions:
    # NOTE:  Function call is optional
    generate_statistics(data)


    # Printing dataframe header (= all imported features & labels):
    print('\n============================= DATAFRAME HEADER ==============================')
    print(data.iloc[0]) # =header



    # III. Further Data Cleaning & Encoding:

    # Filtering out entries (=campaigns) with 'canceled' or 'live' state values (as label). Just using 'successful' and 'failed' campaigns for classification:
    data_filtered = data.loc[data['state'].isin(['successful','failed'])]

    # Encoding string-value columns to logical datatypes (binary & ordinal) for classification:
    data_encoded = feature_encoding(data_filtered)



    # IV. Feature-Selection:

    # a) Manual Feature-Pre-Selection:

    # NOTE:  manually add/remove features in following line forfeature-selection:
    feature_preselection = ['backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'goal_exceeded']#, 'divergence'] #'spotlight'
    # features = ['backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence', 'goal_exceeded']

    data_features = data_encoded[feature_preselection]
    data_labels = data_encoded['state']

    print('\n============================= FEATURE-SELECTION =============================\n')
    print('> Manual Feature-Pre-Selection:')
    for feature in feature_preselection:
        print(' ',feature, end=',')
    print('\n\n> Imported Dataset after Feature-Pre-Selection:\t',data_features.shape)


    # b) Automatic Feature-Selection:

    # Univariate automatic feature selection:

    # Applying SelectKBest class to extract top best features:
    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(data_features, data_labels)
    data_scores = pd.DataFrame(fit.scores_)
    data_columns = pd.DataFrame(data_features.columns)
    # Concat two dataframes for better visualization:
    feature_scores = pd.concat([data_columns, data_scores],axis=1)
    feature_scores.columns = ['Feature','Score']
    feature_scores.sort_values(by=['Score'])

    # Removing features with low variance:
    #remover = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #data_features = remover.fit_transform(data_features)
    #print(data_features.iloc[0]) # =header



    # V. Data division into Training and Test datasets & Normalization:
    ratio = 0.3

    features_test, features_train, labels_test, labels_train = train_test_split(data_features, data_labels, test_size=ratio)

    print('\n============================= DATASET DIVISION ==============================\n')
    print('> split ratio:\t',ratio,'/',1.0-ratio)
    print('> Testset:\t',features_test.shape)
    print('> Trainingset:\t',features_test.shape)
    print('> Full imported Dataset: ',data.shape,'\n')


    # Normalization (L1 & L2):
    # NOTE:  Change 'normtype' value to 'l1' / 'l2' to change normalization type:
    normtype = 'l2'#'l1'
    print('> Normalizing datasets with', normtype,':')
    features_test= normalize(features_test, norm=normtype)
    features_train = normalize(features_train, norm=normtype)



    # VI. Classification:
    print('============================== CLASSIFICATION ===============================\n')

    # cls_selection is used for manually enabeling the individual classifiers.
    # NOTE:  setting boolean value, eanbles/disables classifiers
    classifiers = {
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


    scores, importances = classify(features_test, labels_test, features_train, labels_train, classifiers, feature_preselection)

    importance_ranking = rank_feature_importance(importances)


    print('\n============================== FEATURE RANKING ==============================\n')

    print('> Table containing importance of every feature with different classifiers:\n')
    print(importances.to_string(),'\n')


    print('> Features with highest importance with different classifiers:')
    print(importance_ranking)


    print('\n> Univariate automatic feature selection:\nApplying SelectKBest class to extract top best features:')
    print(feature_scores)
    #print(feature_scores.round({'Score': 3}))
    #print(feature_scores.nlargest(20,'Score', ))  #print n best features


    print('\n============================= CLASSIFIER RANKING =============================\n')
    print(scores)

    print('\n=============================================================================')


if __name__ == '__main__':
    main()
