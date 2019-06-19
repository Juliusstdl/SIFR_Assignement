#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# author:   Julius Steidl
# date:     19.06.2019
# version:  0.1
# note:  directory with .csv files:  ./Kickstarter_2019-04-18T03_20_02_220Z/

directory = 'Kickstarter_2019-04-18T03_20_02_220Z'

import os
import sys
import csv
import pickle
import re
#import numpy as np


# fixing error "_csv.Error: field larger than field limit (131072)"
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


output = open('output.txt','w', encoding="utf8")

all_campaigns = []

for filename in os.listdir(directory):

    # only using first file for now ('Kickstarter.csv'):
    if filename.endswith(".csv") and filename == 'Kickstarter.csv':

        with open(directory+'/'+filename, 'r', encoding="utf8", newline='') as csvfile:

            fieldnames = csvfile.readline().split(',')

            csvreader = csv.reader(csvfile, quotechar='"', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)

            for row in csvreader:
                campaign = {}
                for field, value in zip(fieldnames, row):
                    campaign[field] = value

                all_campaigns.append(campaign)

                continue

        # save as .pickle file  (-> not necessary to parse .csv files first):
        with open(filename+'.pickle', 'wb') as handle:
            pickle.dump(all_campaigns, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        continue

    # load from .pickle file:
    with open(filename+'.pickle', 'rb') as handle:
        all_campaigns = []
        all_campaigns = pickle.load(handle)


    counter = 0
    for campaign in all_campaigns:
        # values = backers_count,blurb,category,converted_pledged_amount,country,created_at,creator,currency,currency_symbol,currency_trailing_code,current_currency,deadline,disable_communication,friends,fx_rate,goal,id,is_backing,is_starrable,is_starred,launched_at,location,name,permissions,photo,pledged,profile,slug,source_url,spotlight,staff_pick,state,state_changed_at,static_usd_rate,urls,usd_pledged,usd_type

        print( str(counter)+': '+campaign['blurb'] )
        print('- state: '+campaign['state'] )
        output.write( str(counter)+': '+str(campaign['blurb'])+'\n- state: '+str(campaign['state'])+'\n' )

        for field, value in campaign.items():

            # filtering some insifignicant fields out:
            if field not in ['blurb', 'state', 'profile', 'photo', 'urls', 'source_url', 'creator', 'location']:
                print( '- '+field+': '+value )
                output.write ('- '+str(field)+': '+str(value)+'\n' )
        print('\n')
        output.write('\n')

        counter += 1

output.close()
