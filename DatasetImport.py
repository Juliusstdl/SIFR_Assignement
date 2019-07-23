#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Dataset Import:
# Importing Dataset & Data Cleaning.


import os
import pandas as pd
import numpy as np
import ast
import re
import operator
from datetime import datetime


class DatasetImport:

    def __init__(self, path):
        self.path = path
        self.all_data = self.import_datasets(path)

    def import_datasets(self, path):
        all_data = pd.DataFrame()

        print('> Importing .csv-files from folder: ',path)
        filecount = 0

        # Iterating over all .csv files in subdirectory:
        for filename in os.listdir(path):

            if filename.endswith(".csv"):
                print(' '+str(filecount)+': '+filename)
                filecount += 1

                with open(path+'/'+filename, 'r', encoding="utf8", newline='') as csvfile:

                    raw_data = pd.read_csv(path+'/'+filename, header=None)

                    # 1. Selecting dataframe columns for import:
                    data = raw_data[[31,16,22,0,3,15,4,30,29,20,11]]

                    # 2. Using first line in .csv-file as index for columns:
                    headers = data.iloc[0]
                    data = pd.DataFrame(data.values[1:], columns=headers)

                    # 3. Coverting columns with string-values in dataframe to numeric- & boolean-values:
                    data['id'] = pd.to_numeric(data['id'])
                    data['backers_count'] = pd.to_numeric(data['backers_count'])
                    data['converted_pledged_amount'] = pd.to_numeric(data['converted_pledged_amount'])
                    data['goal'] = pd.to_numeric(data['goal'])
                    data['launched_at'] = pd.to_numeric(data['launched_at'])
                    data['deadline'] = pd.to_numeric(data['deadline'])

                    bool_dict = {'true': True,'false': False}
                    data['staff_pick'] = data['staff_pick'].map(bool_dict)
                    data['spotlight'] = data['spotlight'].map(bool_dict)


                    # a) Parsing 'category'-column string as individual columns/features:
                    cat_id = []; cat_name = []; subcat_name = []; pos = []; parent_id = []

                    for string in [str(x) for x in raw_data[2][1:]]:

                        cat_mask = re.compile('{"id":(\d+?),.+"slug":"(.+?)(\/(.+?))?","position":(\d+?)(,"parent_id":(\d+?))?,')
                        extract = re.search(cat_mask, string)

                        if extract and extract.group(1) is not None:
                            cat_id.append(int(extract.group(1)))
                        else:
                            cat_id.append(np.NaN)

                        if extract and extract.group(2) is not None:
                            cat_name.append(extract.group(2))
                        else:
                            cat_name.append(np.NaN)

                        if extract and extract.group(4) is not None:
                            subcat_name.append(extract.group(4))
                        else:
                            subcat_name.append(np.NaN)

                        if extract and extract.group(5) is not None:
                            pos.append(int(extract.group(5)))
                        else:
                            pos.append(np.NaN)

                        if extract and extract.group(7) is not None:
                            parent_id.append(int(extract.group(7)))
                        else:
                            parent_id.append(np.NaN)


                    # b) Parsing 'creator'-column string as individual columns/features:
                    person_id = []; person_name = [];

                    for string in [str(x) for x in raw_data[6][1:]]:
                        string = str(string)

                        person_mask = re.compile('{"id":(\d+?),"name":"(.+?)"')
                        extract = re.search(person_mask, string)

                        if extract and extract.group(1) is not None:
                            person_id.append(int(extract.group(1)))
                        else:
                            person_id.append(np.NaN)

                        if extract and extract.group(2) is not None:
                            person_name.append(extract.group(2))
                        else:
                            person_name.append(np.NaN)


                    # Parsing 'location'-column string as individual columns/features:
                    location_id = []; location_name = []; location_state = []; location_type = []

                    for string in [str(x) for x in raw_data[21][1:]]:

                        location_mask = re.compile('{"id":(\d+?),.+"short_name":"(.+?)".+"state":"(.+?)","type":"(.+?)"')
                        extract = re.search(location_mask, string)

                        if extract and extract.group(1) is not None:
                            location_id.append(int(extract.group(1)))
                        else:
                            location_id.append(np.NaN)

                        if extract and extract.group(2) is not None:
                            location_name.append(extract.group(2))
                        else:
                            location_name.append(np.NaN)

                        if extract and extract.group(3) is not None:
                            location_state.append(extract.group(3))
                        else:
                            location_state.append(np.NaN)

                        if extract and extract.group(4) is not None:
                            location_type.append(extract.group(4))
                        else:
                            location_type.append(np.NaN)


                    # c) Calculating campaign duration_days:
                    duration_days = []; divergence = []; goal_exceeded = []

                    for index, row in data.iterrows():
                        launched_at = row['launched_at']
                        deadline = row['deadline']
                        # Calculating time differece in days:
                        difference = int(round((float(deadline)-float(launched_at))/(60*60*24)))
                        duration_days.append(difference)
                        #print('launched at:',datetime.utcfromtimestamp(launched_at).strftime('%d-%m-%y'), '- deadline:',datetime.utcfromtimestamp(deadline).strftime('%d-%m-%y'), '- difference in days:',difference)

                        # Calculating divergence between pledged money and goal:
                        # divergence measeures the divergence value of pledged from goal in USD and can be positive or negative
                        # goal_exceeded is True if pledged money exceeds the goal

                        div = float(row['converted_pledged_amount']) - float(row['goal'])
                        divergence.append(div)

                        if div >= 0:
                            goal_exceeded.append(True)
                        else:
                            goal_exceeded.append(False)


                # Assigning all in previous steps a), b), c) retrieved data fields to dataframe:
                data = data.assign( cat_id = cat_id, cat_name = cat_name, subcat_name = subcat_name, pos = pos, parent_id = parent_id, person_id = person_id, person_name = person_name, location_id = location_id, location_name = location_name, location_state = location_state, location_type = location_type, duration_days = duration_days, divergence = divergence, goal_exceeded = goal_exceeded)

                # Rearranging column sequence:
                columns = ['state', 'id', 'name', 'backers_count', 'converted_pledged_amount', 'goal', 'country', 'staff_pick', 'spotlight', 'launched_at', 'deadline', 'cat_id', 'cat_name', 'subcat_name', 'pos', 'parent_id', 'person_id', 'person_name', 'location_id', 'location_name', 'location_state', 'location_type', 'duration_days', 'divergence', 'goal_exceeded']

                data = data[columns]
                all_data = all_data.append(data)
                #all_data = data

            else:
                continue

        return all_data
