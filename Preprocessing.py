#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Preprocessing:
# Encoding string-value columns to logical datatypes (binary & ordinal) for estimation.


import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Preprocessing:

    def __init__(self, data):
        self.data = data
        self.data_encoded = self.feature_encoding(data)


    def feature_encoding(self, data):
        labelencoder = LabelEncoder()

        # Converting string values from 'state'-column into boolean values:
        bool_dict = {'successful': True,'failed': False}
        data['state'] = data['state'].map(bool_dict)

        # Fill string fields containing NaN dummy-value with 'NaN' string required for labelencoder:
        values = {'subcat_name': 'NaN', 'location_name': 'NaN', 'location_state': 'NaN', 'location_type': 'NaN'}
        data = data.fillna(value=values)

        data['country'] = labelencoder.fit_transform(data['country'])
        data['cat_name'] = labelencoder.fit_transform(data['cat_name'])
        data['subcat_name'] = labelencoder.fit_transform(data['subcat_name'])
        data['person_name'] = labelencoder.fit_transform(data['person_name'])
        data['location_name'] = labelencoder.fit_transform(data['location_name'])
        data['location_state'] = labelencoder.fit_transform(data['location_state'])
        data['location_type'] = labelencoder.fit_transform(data['location_type'])
        data['year'] = labelencoder.fit_transform(data['year'])
        data['month'] = labelencoder.fit_transform(data['month'])


        data['cat_id'] = labelencoder.fit_transform(data['cat_id'])
        data['parent_id'] = labelencoder.fit_transform(data['parent_id'])
        data['person_id'] = labelencoder.fit_transform(data['person_id'])
        data['location_id'] = labelencoder.fit_transform(data['location_id'])
        #data['duration_days'] = labelencoder.fit_transform(data['duration_days'])

        # Fill numeric fields containing NaN dummy-value with 0 as value required for fiting model:
        data = data.fillna(0)

        return data
