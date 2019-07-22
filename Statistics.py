#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Generate Statistics:
# Generating some statistics for getting insight into dataset & for supporting manual feature-selection decisions.


class Statistics:

    def __init__(self, data):
        self.data = data
        self.generate_statistics(data)


    def generate_statistics(self, data):

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
