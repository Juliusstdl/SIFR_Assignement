#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# SFIR assignement: predicting campaign success on kickstarter using ML
# author:   Julius Steidl
# date:     28.06.2019
# version:  0.4
# note:  directory with .csv files:  ./Kickstarter_2019-04-18T03_20_02_220Z/


import os
import sys
import csv
import pickle
import re
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter
#from sklearn import numpy


# fixing error "_csv.Error: field larger than field limit (131072)"
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)



def import_datasets(directory):

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

            # save as .pickle file  (-> not necessary to parse .csv files every time):
            with open(filename+'.pickle', 'wb') as handle:
                pickle.dump(all_campaigns, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            continue

    return all_campaigns



def generate_statistics(all_campaigns):

    campaign_count = 0
    success_count = 0
    backers_success = 0
    pledged_success = 0
    goal_success = 0
    spot_success = 0
    pick_success = 0
    fail_count = 0
    backers_fail = 0
    pledged_fail = 0
    goal_fail = 0
    spot_fail = 0
    pick_fail = 0
    cancel_count = 0
    backers_cancel = 0
    pledged_cancel = 0
    goal_cancel = 0
    spot_cancel = 0
    pick_cancel = 0

    country_success = defaultdict(int)
    country_fail = defaultdict(int)
    country_cancel = defaultdict(int)

    cat_success = defaultdict(int)
    cat_fail = defaultdict(int)
    cat_cancel = defaultdict(int)
    subcat_success = defaultdict(int)
    subcat_fail = defaultdict(int)
    subcat_cancel = defaultdict(int)

    for campaign in all_campaigns:

        # values = backers_count,blurb,category,converted_pledged_amount,country,created_at,creator,currency,currency_symbol,currency_trailing_code,current_currency,deadline,disable_communication,friends,fx_rate,goal,id,is_backing,is_starrable,is_starred,launched_at,location,name,permissions,photo,pledged,profile,slug,source_url,spotlight,staff_pick,state,state_changed_at,static_usd_rate,urls,usd_pledged,usd_type
        state = campaign['state']
        backers = campaign['backers_count']
        pledged = campaign['usd_pledged']
        goal = campaign['goal']
        spotlight = campaign['spotlight']
        staff_pick = campaign['staff_pick']
        country = str(campaign['country'])

        extract = re.search('\"slug\"\:\"(.+?)\/(.+?)\",\"', str(campaign['category']))
        if extract:
            category = extract.group(1)
            subcat = category+'\t'+extract.group(2)
        else:
            category = 'default'; subcat = 'default/default'


        if state == 'successful':
            success_count += 1
            backers_success = backers_success + float(backers)
            pledged_success = pledged_success + float(pledged)
            goal_success = goal_success + float(goal)
            if staff_pick == 'true':
                pick_success += 1
            if spotlight == 'true':
                spot_success += 1
            country_success[country] += 1
            cat_success[category] += 1
            subcat_success[subcat] += 1

        elif state == 'failed':
            fail_count += 1
            backers_fail = backers_fail + float(backers)
            pledged_fail = pledged_fail + float(pledged)
            goal_fail = goal_fail + float(goal)
            if staff_pick == 'true':
                pick_fail += 1
            if spotlight == 'true':
                spot_fail += 1
            country_fail[country] += 1
            cat_fail[category] += 1
            subcat_fail[subcat] += 1

        elif state == 'canceled':
            cancel_count += 1
            backers_cancel = backers_cancel + float(backers)
            pledged_cancel = pledged_cancel + float(pledged)
            goal_cancel = goal_cancel + float(goal)
            if staff_pick == 'true':
                pick_cancel += 1
            if spotlight == 'true':
                spot_cancel += 1
            country_cancel[country] += 1
            cat_cancel[category] += 1
            subcat_cancel[subcat] += 1

        print('\n',campaign_count,': ',campaign['name'],': ',campaign['blurb'])
        print('- state: ',state)
        print('- backers: ',backers)
        print('- pledged: ',pledged)
        print('- goal: ',goal)
        print('- country: ',country)
        print('- category: ',category,'\tsubcategory: ',subcat)

        for field, value in campaign.items():

            # filtering some insifignicant fields out:
            if field not in ['name', 'blurb', 'state', 'backers', 'pledged', 'goal', 'country', 'category', 'profile', 'photo', 'urls', 'source_url', 'creator', 'location']:
                print('-',field,': ',value)

        campaign_count += 1

    print('\n========== EXPLORATORY DATA ANALYSIS ==========\n')

    print('> Total number of campaigns: ',campaign_count)
    print('\t- backers avg.:\t',int(float((backers_success+backers_fail+backers_cancel)/float(campaign_count))))
    print('\t- pledged avg.:\t',int(float((pledged_success+pledged_cancel+pledged_cancel)/float(campaign_count))),' $')
    print('\t- goal avg.:\t',int(float((goal_success+goal_fail+goal_cancel)/float(campaign_count))),' $')
    print('\t- staff pick:\t',int(100 * float(pick_success+pick_fail+pick_cancel)/float(campaign_count)),' %')
    print('\t- spotlight:\t',int(100 * float(spot_success+spot_fail+spot_cancel)/float(campaign_count)),' %\n\n')

    print('> successful:  ',int(100 * float(success_count)/float(campaign_count)),' %')
    print('\t- backers avg.:\t',int(float(backers_success/float(success_count))))
    print('\t- pledged avg.:\t',int(float(pledged_success/float(success_count))),' $')
    print('\t- goal avg.:\t',int(float(goal_success/float(success_count))),' $')
    print('\t- staff pick:\t',int(100 * float(pick_success)/float(success_count)),' %')
    print('\t- spotlight:\t',int(100 * float(spot_success)/float(success_count)),' %\n')

    print('> failed:  ',int(100 * float(fail_count)/float(campaign_count)),' %')
    print('\t- backers avg.:\t',int(float(backers_fail/float(fail_count))))
    print('\t- pledged avg.:\t',int(float(pledged_fail/float(fail_count))),' $')
    print('\t- goal avg.:\t',int(float(goal_fail/float(fail_count))),' $')
    print('\t- staff pick:\t',int(100 * float(pick_fail)/float(fail_count)),' %')
    print('\t- spotlight:\t',int(100 * float(spot_fail)/float(fail_count)),' %\n')

    print('> canceled:  ',int(100 * float(cancel_count)/float(campaign_count)),' %')
    print('\t- backers avg.:\t',int(float(backers_cancel/float(cancel_count))))
    print('\t- pledged avg.:\t',int(float(pledged_cancel/float(cancel_count))),' $')
    print('\t- goal avg.:\t',int(float(goal_cancel/float(cancel_count))),' $')
    print('\t- staff pick:\t',int(100 * float(pick_cancel)/float(cancel_count)),' %')
    print('\t- spotlight:\t',int(100 * float(spot_cancel)/float(cancel_count)),' %\n')

    print('\n> countries with most successful campaigns:')
    top_country_success = OrderedDict(sorted(country_success.items(), key=itemgetter(1), reverse=True))
    for country, value in top_country_success.items():
        print('\t',country,'-',value)

    print('\n> countries with most failed campaigns:')
    top_country_fail = OrderedDict(sorted(country_fail.items(), key=itemgetter(1), reverse=True))
    for country, value in top_country_fail.items():
        print('\t',country,'-',value)

    print('\n> countries with most canceled campaigns:')
    top_country_cancel = OrderedDict(sorted(country_cancel.items(), key=itemgetter(1), reverse=True))
    for country, value in top_country_cancel.items():
        print('\t',country,'-',value)


    print('\n> categories with most successful campaigns:')
    top_cat_success = OrderedDict(sorted(cat_success.items(), key=itemgetter(1), reverse=True))
    for cat, value in top_cat_success.items():
        print('\t',cat,'-',value)

    print('\n> subcategories with most successful campaigns:')
    top_subcat_success = OrderedDict(sorted(subcat_success.items(), key=itemgetter(1), reverse=True))
    for subcat, value in top_subcat_success.items():
        print('\t',subcat,'-',value)

    print('\n> categories with most failed campaigns:')
    top_cat_fail = OrderedDict(sorted(cat_fail.items(), key=itemgetter(1), reverse=True))
    for cat, value in top_cat_fail.items():
        print('\t',cat,'-',value)

    print('\n> subcategories with most failed campaigns:')
    top_subcat_fail = OrderedDict(sorted(subcat_fail.items(), key=itemgetter(1), reverse=True))
    for subcat, value in top_subcat_fail.items():
        print('\t',subcat,'-',value)

    print('\n===============================================')



directory = 'Kickstarter_2019-04-18T03_20_02_220Z'  # date: 2019-05-16

all_campaigns = import_datasets(directory)


# load from .pickle file:
filename = 'Kickstarter.csv'
with open(filename+'.pickle', 'rb') as handle:
    all_campaigns = []
    all_campaigns = pickle.load(handle)


generate_statistics(all_campaigns)
