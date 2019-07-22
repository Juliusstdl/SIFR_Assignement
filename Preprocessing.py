#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Preprocessing:
# Encoding string-value columns to logical datatypes (binary & ordinal) for estimation.


from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Preprocessing:

    def __init__(self, data):
        self.data = data
        self.data_encoded = self.feature_encoding(data)


    def feature_encoding(self, data):
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
