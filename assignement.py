#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# author:   Julius Steidl
# date:     06.07.2019
# version:  1.0
# note:     directory with .csv files:  ./Kickstarter_2019-04-18T03_20_02_220Z/

import os
import pandas as pd
import numpy as np
import ast
import re
import operator
from datetime import datetime
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def import_datasets(directory):

    # Iterating over all .csv files in subdirectory:
    for filename in os.listdir(directory):

        # only using first file for now ('Kickstarter.csv'):
        if filename.endswith(".csv") and filename == 'Kickstarter.csv':

            with open(directory+'/'+filename, 'r', encoding="utf8", newline='') as csvfile:

                raw_data = pd.read_csv('Kickstarter_2019-04-18T03_20_02_220Z/'+filename, header=None)

                data = raw_data[[31,16,22,0,3,15,4,30,29,20,11]]

                headers = data.iloc[0]

                data = pd.DataFrame(data.values[1:], columns=headers)


                # Parsing 'category'-column strings as individual columns/features:
                cat_id = []; name_cat = []; cat = []; subcat = []; pos = []; parent_id = []

                for string in raw_data[2].astype(str):

                    cat_mask = re.compile('{"id":(\d+?),"name":"(.+?)","slug":"(.+?)(\/(.+?))?","position":(\d+?)(,"parent_id":(\d+?))?,')
                    extract = re.search(cat_mask, string)

                    if extract:
                        if extract.group(1) is not None:
                            cat_id.append(int(extract.group(1)))
                        else:
                            cat_id.append(0)

                        if extract.group(2) is not None:
                            name_cat.append(extract.group(2))
                        else:
                            name_cat.append('default')

                        if extract.group(3) is not None:
                            cat.append(extract.group(3))
                        else:
                            cat.append('default')

                        if extract.group(5) is not None:
                            subcat.append(extract.group(5))
                        else:
                            subcat.append('default')

                        if extract.group(6) is not None:
                            pos.append(int(extract.group(6)))
                        else:
                            pos.append(0)

                        if extract.group(8) is not None:
                            parent_id.append(int(extract.group(8)))
                        else:
                            parent_id.append(0)


                # Parsing 'creator'-column strings as individual columns/features:
                person_id = []; person_name = []; username = [];

                for string in raw_data[6].astype(str):

                    person_mask = re.compile('{"id":(\d+?),"name":"(.+?)"')
                    extract = re.search(person_mask, string)

                    if extract:
                        if extract.group(1) is not None:
                            person_id.append(int(extract.group(1)))
                        else:
                            person_id.append(0)

                        if extract.group(2) is not None:
                            person_name.append(extract.group(2))
                        else:
                            person_name.append('default')


                # Calculating campaign duration (duration = 'deadline' - 'launched_at'):
                duration = []

                for index, row in data.iterrows():
                    launched_at = row['launched_at']
                    deadline = row['deadline']
                    difference = int(round((float(deadline)-float(launched_at))/(60*60*24)))
                    duration.append(difference)
                    #print('launched at:',datetime.utcfromtimestamp(launched_at).strftime('%d-%m-%Y'), '- deadline:',datetime.utcfromtimestamp(deadline).strftime('%d-%m-%Y'), '- difference in days:',difference)


            data = data.assign( cat_id = cat_id, name_cat = name_cat, category = cat, subcategory = subcat, pos = pos, parent_id = parent_id, person_id = person_id, person_name = person_name, duration = duration )

            # Coverting string-values in dataframe to numeric- & boolean-values:
            data['id'] = pd.to_numeric(data['id'])
            data['backers_count'] = pd.to_numeric(data['backers_count'])
            data['converted_pledged_amount'] = pd.to_numeric(data['converted_pledged_amount'])
            data['goal'] = pd.to_numeric(data['goal'])

            bool_dict = {'true': True,'false': False}
            data['staff_pick'] = data['staff_pick'].map(bool_dict)
            data['spotlight'] = data['spotlight'].map(bool_dict)

            # Rearranging column sequence:
            cols = ['state', 'name', 'id', 'backers_count', 'converted_pledged_amount', 'goal', 'staff_pick', 'spotlight', 'country', 'category', 'subcategory', 'pos', 'cat_id', 'parent_id', 'name_cat', 'person_name', 'person_id', 'duration', 'launched_at', 'deadline']
            data = data[cols]

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

    print('\n> countries with most successful campaigns:')
    print(successful.country.value_counts()[:top])
    print('\n> countries with most unsuccessful campaigns:')
    print(unsuccessful.country.value_counts()[:top])

    print('\n> categories with most successful campaigns:')
    print(successful.category.value_counts()[:top])
    print('\n> categories with most unsuccessful campaigns:')
    print(unsuccessful.category.value_counts()[:top])

    print('\n> subcategories with most successful campaigns:')
    print(successful.subcategory.value_counts()[:top])
    print('\n> subcategories with most unsuccessful campaigns:')
    print(unsuccessful.subcategory.value_counts()[:top])

    print('\n> persons with most successful campaigns:')
    print(successful.person_name.value_counts()[:top])
    print('\n> persons with most unsuccessful campaigns:')
    print(unsuccessful.person_name.value_counts()[:top])


def classify(features_test, labels_test, features_train, labels_train):
    scores = {}

    print('\n\n============================== CLASSIFICATION ===============================\n')

    # A) SVC:
    cls_SVC = SVC(gamma='auto') #(gamma=2, C=1), (kernel="linear", C=0.025)
    cls_SVC.fit(features_train, labels_train)
    score_SVC = cls_SVC.score(features_test, labels_test)
    scores['SVC'] = score_SVC

    print('>',cls_SVC)
    print('> Score:  ',score_SVC,'\n\n')


    # B) KNeighbors:
    cls_KNB = KNeighborsClassifier(n_neighbors=4) #3
    cls_KNB.fit(features_train, labels_train)
    score_KNB = cls_KNB.score(features_test, labels_test)
    scores['KNeighbors'] = score_KNB

    print('>',cls_KNB)
    print('> Score:  ',score_KNB,'\n\n')


    # C) GaussianProcess:
    cls_GPC = GaussianProcessClassifier()#(1.0 * RBF(1.0))
    cls_GPC.fit(features_train, labels_train)
    score_GPC = cls_GPC.score(features_test, labels_test)
    scores['GaussianProcess'] = score_GPC

    print('>',cls_GPC)
    print('> Score:  ',score_GPC,'\n\n')


    # D) DecisionTree:
    cls_DCT = DecisionTreeClassifier(max_depth=5)
    cls_DCT.fit(features_train, labels_train)
    score_DCT = cls_DCT.score(features_test, labels_test)
    scores['DecisionTree'] = score_DCT

    print('>',cls_DCT)
    print('> Score:  ',score_DCT,'\n\n')


    # E) GaussianNB:
    cls_GNB = GaussianNB()
    cls_GNB.fit(features_train, labels_train)
    score_GNB = cls_GNB.score(features_test, labels_test)
    scores['GaussianNB'] = score_GNB

    print('>',cls_GNB)
    print('> Score:  ',score_GNB,'\n\n')


    # F) MLP:
    cls_MLP = MLPClassifier(alpha=1, max_iter=1000)
    cls_MLP.fit(features_train, labels_train)
    score_MLP = cls_MLP.score(features_test, labels_test)
    scores['MLP'] = score_MLP

    print('>',cls_MLP)
    print('> Score:  ',score_MLP,'\n\n')


    return scores



# I.-II. Importing Dataset & Data Cleaning:

directory = 'Kickstarter_2019-04-18T03_20_02_220Z'  # date: 2019-05-16

# saving / loading dataframe as .pickle file  (-> not necessary to parse .csv-files every time):
# NOTE:  uncomment next two lines for skipping import of new dataset from .csv-files:
data_new = import_datasets(directory)
data_new.to_pickle(directory+'.pickle')

data = pd.read_pickle('./'+directory+'.pickle')



# III. Exploratory data analysis:
# Generating some statistics for getting insight into dataset & for supporting manual feature-selection decisions:
generate_statistics(data)



# Printing dataframe header (= all imported features & labels):
headers = data.iloc[0]

print('\n============================= DATAFRAME HEADER ==============================')
print(headers)



# Filtering out entries (=campaigns) with 'canceled' or 'live' state values:
data = data.loc[data['state'].isin(['successful','failed'])]

# Converting string values from 'state'-column into binary integers (0 / 1):
bool_dict = {'successful': 1,'failed': 0}
data['state'] = data['state'].map(bool_dict)

# Converting boolean values from 'staff_pick' & 'spotlight'-column into binary integers (0 / 1):
#bool_dict2 = {True: 1, True: 0}
#data['staff_pick'] = data['staff_pick'].map(bool_dict2)
#data['spotlight'] = data['spotlight'].map(bool_dict2)



# IV. Data division into Training and Test datasets
ratio = 0.3

test_set, train_set = train_test_split(data, test_size=ratio)

print('\n============================= DATASET DIVISION ==============================\n')
print('> split ratio:\t',ratio,'/',1.0-ratio)
print('> Testset:\t',test_set.shape)
print('> Trainingset:\t',train_set.shape)
print('> Full Dataset:\t',data.shape)



# V. Feature-Selection:

# NOTE:  manually add/remove features in following line forfeature-selection:
feature_selection = ['backers_count', 'converted_pledged_amount', 'staff_pick']#, 'goal']
# features = ['name', 'id', 'backers_count', 'converted_pledged_amount', 'goal', 'staff_pick', 'spotlight', 'country', 'category', 'subcategory', 'pos', 'cat_id', 'parent_id', 'name_cat', 'person_name', 'person_id', 'duration', 'launched_at', 'deadline']

features_test = test_set[feature_selection]
labels_test = test_set['state'] # labels = 'state'

features_train = train_set[feature_selection]
labels_train = train_set['state']

print('\n============================= FEATURE-SELECTION =============================\n')
print('> Selected features:')
for feature in feature_selection:
    print(' ',feature, end=',')



# VI. Classification:
scores = classify(features_test, labels_test, features_train, labels_train)

top_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

print('============================= CLASSIFIER RATING =============================\n')
i = 1
for cls, score in top_scores:
    print('  '+str(i)+'. '+str(cls)+':\t'+str(score))
    i += 1

print('\n=============================================================================')
