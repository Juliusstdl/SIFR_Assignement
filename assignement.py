#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# author:   Julius Steidl
# date:     16.07.2019
# version:  1.3
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

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

pd.options.mode.chained_assignment = None  # default='warn'


def import_datasets(directory, number_of_files):
    print('> Importing',number_of_files,'.csv-files from folder: ',directory)
    filecount = 0

    # Iterating over all .csv files in subdirectory:
    for filename in os.listdir(directory):

        if filename.endswith(".csv") and filecount < number_of_files:
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
                duration = []; divergence = []

                for index, row in data.iterrows():
                    launched_at = row['launched_at']
                    deadline = row['deadline']
                    difference = int(round((float(deadline)-float(launched_at))/(60*60*24)))
                    duration.append(difference)
                    #print('launched at:',datetime.utcfromtimestamp(launched_at).strftime('%d-%m-%Y'), '- deadline:',datetime.utcfromtimestamp(deadline).strftime('%d-%m-%Y'), '- difference in days:',difference)

                    # Calculating divergence between pledged money and goal:
                    divergence.append(float(row['converted_pledged_amount'])-float(row['goal']))


            data = data.assign( cat_id = cat_id, cat_name = cat_name, subcat_name = subcat_name, pos = pos, parent_id = parent_id, person_id = person_id, person_name = person_name, location_id = location_id, location_name = location_name, location_state = location_state, location_type = location_type, duration = duration , divergence = divergence)
            # Coverting string-values in dataframe to numeric- & boolean-values:
            data['id'] = pd.to_numeric(data['id'])
            data['backers_count'] = pd.to_numeric(data['backers_count'])
            data['converted_pledged_amount'] = pd.to_numeric(data['converted_pledged_amount'])
            data['goal'] = pd.to_numeric(data['goal'])

            bool_dict = {'true': True,'false': False}
            data['staff_pick'] = data['staff_pick'].map(bool_dict)
            data['spotlight'] = data['spotlight'].map(bool_dict)

            # Rearranging column sequence:
            columns = ['state', 'id', 'name', 'backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence']

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

    # data = ['state', 'id', 'name', 'backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence']

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


def classify(features_test, labels_test, features_train, labels_train, feature_selection, cls_selection):

    scores = {}
    importances = pd.DataFrame(columns=feature_selection)


    # A) Extra Trees:
    ETC, name_ETC = cls_selection['ETC']
    if ETC:
        cls_ETC = ExtraTreesClassifier()
        cls_ETC.fit(features_train, labels_train)
        score_ETC = cls_ETC.score(features_test, labels_test)
        importances_ETC = get_feature_importances(cls_ETC, feature_selection)
        scores[name_ETC] = score_ETC
        importances.loc[name_ETC] = importances_ETC


    # B) Random Forest:
    RFC, name_RFC = cls_selection['RFC']
    if RFC:
        cls_RFC = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
        cls_RFC.fit(features_train, labels_train)
        score_RFC = cls_RFC.score(features_test, labels_test)
        importances_RFC = get_feature_importances(cls_RFC, feature_selection)
        scores[name_RFC] = score_RFC
        importances.loc[name_RFC] = importances_RFC


    # C) AdaBoost:
    ABC, name_ABC = cls_selection['ABC']
    if ABC:
        cls_ABC = AdaBoostClassifier()
        cls_ABC.fit(features_train, labels_train)
        score_ABC = cls_ABC.score(features_test, labels_test)
        importances_ABC = get_feature_importances(cls_ABC, feature_selection)
        scores[name_ABC] = score_ABC
        importances.loc[name_ABC] = importances_ABC


    # D) Decision Tree:
    DCT, name_DCT = cls_selection['DCT']
    if DCT:
        cls_DCT = DecisionTreeClassifier(max_depth=5)
        cls_DCT.fit(features_train, labels_train)
        score_DCT = cls_DCT.score(features_test, labels_test)
        importances_DCT = get_feature_importances(cls_DCT, feature_selection)
        scores[name_DCT] = score_DCT
        importances.loc[name_DCT] = importances_DCT


    # E) KNearestNeighbors:
    KNB, name_KNB = cls_selection['KNB']
    if KNB:
        cls_KNB = KNeighborsClassifier(n_neighbors=4) #3
        cls_KNB.fit(features_train, labels_train)
        score_KNB = cls_KNB.score(features_test, labels_test)
        scores[name_KNB] = score_KNB


    # F) Linear SVC:
    LSV, name_LSV = cls_selection['LSV']
    if LSV:
        cls_LSV = SVC(kernel="linear", C=0.025) # C=0.01, penalty="l1", dual=False)
        cls_LSV.fit(features_train, labels_train)
        score_LSV = cls_LSV.score(features_test, labels_test)
        scores[name_LSV] = score_LSV

        # Feature selection using SelectFromModel:
        #cls_LSV = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data_features, data_labels)
        #model = SelectFromModel(cls_LSV, prefit=True)
        #features_train_new = model.transform(features_train)


    # G) RBF SVC:
    RBF, name_RBF = cls_selection['RBF']
    if RBF:
        cls_RBF = SVC(gamma=2, C=1) # gamma='auto')
        cls_RBF.fit(features_train, labels_train)
        score_RBF = cls_RBF.score(features_test, labels_test)
        scores[name_RBF] = score_RBF


    # H) Gaussian Process:
    GPC, name_GPC = cls_selection['GPC']
    if GPC:
        cls_GPC = GaussianProcessClassifier()#(1.0 * RBF(1.0))
        cls_GPC.fit(features_train, labels_train)
        score_GPC = cls_GPC.score(features_test, labels_test)
        scores[name_GPC] = score_GPC


    # I) Neural Network (MLP):
    MLP, name_MLP = cls_selection['MLP']
    if MLP:
        cls_MLP = MLPClassifier(alpha=1, max_iter=1000)
        cls_MLP.fit(features_train, labels_train)
        score_MLP = cls_MLP.score(features_test, labels_test)
        scores[name_MLP] = score_MLP


    # J) Logistic Regression:
    LRC, name_LRC = cls_selection['LRC']
    if LRC:
        cls_LRC = LogisticRegression()
        cls_LRC.fit(features_train, labels_train)
        score_LRC = cls_LRC.score(features_test, labels_test)
        scores[name_LRC] = score_LRC


    # J) Quadratic Discriminant Analysis (QDA):
    QDA, name_QDA = cls_selection['QDA']
    if QDA:
        cls_QDA = QuadraticDiscriminantAnalysis()
        cls_QDA.fit(features_train, labels_train)
        score_QDA = cls_QDA.score(features_test, labels_test)
        scores[name_QDA] = score_QDA


    # K) Gaussian Naive Bayes:
    GNB, name_GNB = cls_selection['GNB']
    if GNB:
        cls_GNB = GaussianNB()
        cls_GNB.fit(features_train, labels_train)
        score_GNB = cls_GNB.score(features_test, labels_test)
        scores[name_GNB] = score_GNB


    scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    return (scores, importances)


def get_feature_importances(classifier, feature_selection):
    feature_importance = {}

    importances = classifier.feature_importances_
    indices = np.argsort(importances)

    for i, value in zip(indices, importances):
        feature_importance[feature_selection[i]] = value

    #feature_importance = sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True)

    return feature_importance


def rank_features(importances):
    importances.loc['Total',:]= importances.sum(axis=0)

    feature_ranking = (importances.iloc[-1:].reset_index(drop=True)).to_dict()
    for feature, value in feature_ranking.items():
        feature_ranking[feature] = value.get(0)

    feature_ranking = sorted(feature_ranking.items(), key=operator.itemgetter(1), reverse=True)

    return feature_ranking



def main():

    # I. Importing Dataset & Data Cleaning:

    directory = 'Kickstarter_2019-04-18T03_20_02_220Z'  # date: 2019-05-16

    # NOTE:  set value of .csv-files beeing imported from the directory:
    number_of_files = 25

    # saving / loading dataframe as .pickle file  (-> not necessary to parse .csv-files every time):
    # NOTE:  uncomment next two lines for skipping import of new dataset from .csv-files:
    data_imported = import_datasets(directory, number_of_files)
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
    feature_preselection = ['backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration']#, 'divergence'] #'spotlight'
    # features = ['backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration', 'divergence']

    data_features = data_encoded[feature_preselection]
    data_labels = data_encoded['state']

    print('\n============================= FEATURE-SELECTION =============================\n')
    print('> Manual Feature-Pre-Selection:')
    for feature in feature_preselection:
        print(' ',feature, end=',')
    print('\n\n> Imported Dataset after Feature-Pre-Selection:\t',data_features.shape)


    # b) Automatic Feature-Selection:

    # Univariate automatic feature selection:

    # applying SelectKBest class to extract top 10 best features:
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(data_features, data_labels)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data_features.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns


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
    print('> Full imported Dataset:\t',data.shape,'\n')


    # Normalization (L1 & L2):
    features_test= normalize(features_test, norm='l2') # norm='l1'
    features_train = normalize(features_train, norm='l2') # norm='l1'



    # VI. Classification:
    print('============================== CLASSIFICATION ===============================\n')

    # cls_selection is used for manually enabeling the individual classifiers.
    # NOTE:  setting boolean value, eanbles/disables classifiers
    cls_selection = {
    'ETC': (True, 'ExtraTrees'),
    'RFC': (True, 'RandomForest'),
    'ABC': (True, 'AdaBoost'),
    'DCT': (True, 'DecisionTree'),
    'KNB': (True, 'NearestNeighbors'),
    'LSV': (True, 'LinearSVC'),
    'RBF': (True, 'RBF_SVC'),
    'GPC': (True, 'GaussianProcess'),
    'MLP': (True, 'NeuralNetwork'),
    'LRC': (True, 'LogisticRegression'),
    'QDA': (True, 'QuadraticDiscriminantAnalysis'),
    'GNB': (True, 'NaiveBayes') }


    scores, importances = classify(features_test, labels_test, features_train, labels_train, feature_preselection, cls_selection)

    feature_ranking = rank_features(importances)


    print('\n============================= CLASSIFIER RANKING =============================\n')
    i = 1
    for cls, score in scores:
        print('  '+str(i)+'. '+str(cls)+':\t\t\t'+str(score))
        i += 1

    print('\n============================== FEATURE RANKING ==============================\n')

    print('> Table containing importance of every feature with different classifiers:\n')
    print(importances.to_string(),'\n')

    print('> Features with highest importance with different classifiers:')
    i = 1
    for feature, importance in feature_ranking:
        print('  '+str(i)+'. '+str(feature)+':\t\t\t'+str(importance))
        i += 1

    print('\n> Univariate automatic feature selection:\n  Applying SelectKBest class to extract top best features:')
    print(featureScores.nlargest(19,'Score'))  #print n best features

    print('\n=============================================================================')


if __name__ == '__main__':
    main()
