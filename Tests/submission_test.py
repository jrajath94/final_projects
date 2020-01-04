#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
import pandas as pd
import dask.dataframe as ddf
import dask.multiprocessing
import warnings
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from pygeodesy.ellipsoidalVincenty import LatLon
from pygeodesy.utily import m2km
from numba import jit
from Latest_Submission import *


class Olympics_analysis:

    def oly_medal(olympics):
        olympics['Medal'].fillna('No_Medal', inplace=True)
        return olympics

    def one_hot_encoding(olympics):
        olympics["Bronze_Medal"] = np.where(
            olympics["Medal"].str.contains("Bronze"), 1, 0)
        olympics["Silver_Medal"] = np.where(
            olympics["Medal"].str.contains("Silver"), 1, 0)
        olympics["Gold_Medal"] = np.where(
            olympics["Medal"].str.contains("Gold"), 1, 0)
        olympics["No_Medal"] = np.where(
            olympics["Medal"].str.contains("No_Medal"), 1, 0)
        return olympics

    def world_gdp(world_gdp):
        world = world_gdp.drop(['Indicator Name', 'Indicator Code'], axis=1)
        return world

    def noc_merge(olympics, noc_country):
        olympics_NOC = olympics.merge(
            noc_country,
            left_on='NOC',
            right_on='NOC',
            how='left')
        return olympics_NOC

    def olympics_host_years(olympics_host):
        olympics_host['Year'].fillna(-999, inplace=True)
        olympics_host['Winter'].fillna("No_Olympics", inplace=True)
        olympics_host['Summer (Olympiad)'].fillna("No_Olympics", inplace=True)
        olympic_years = list(olympics_host["Year"].unique())
        olympic_years = [str(int(element)) for element in olympic_years]
        return olympics_host, olympic_years

    @jit
    def filter_years(dataframe, world_population_columns, olympic_years):
        for item in world_population_columns:
            if item not in olympic_years:
                dataframe.drop(item, axis=1, inplace=True)
        return dataframe

    def world_population_filter(world_population, olympic_years, olympics_NOC):
        world_population_columns = list(world_population.columns.values)[4:]

        world_population = filter_years(
            world_population,
            world_population_columns,
            olympic_years)
        olympics_NOC_population = olympics_NOC.merge(
            world_population,
            left_on='NOC',
            right_on='Country Code',
            how='left').reset_index(
                drop=True)
        return olympics_NOC_population

    def world_gdp_filter(world_gdp, olympics_NOC, olympic_years):

        world_gdp_columns = list(world_gdp.columns.values)[4:]

        filter_years(world_gdp, world_gdp_columns, olympic_years)

        olympics_NOC_gdp = olympics_NOC.merge(
            world_population,
            left_on='NOC',
            right_on='Country Code',
            how='left').reset_index(
                drop=True)
        return olympics_NOC_gdp

    def team(olympics_NOC):
        gold_medals = olympics_NOC[olympics_NOC.Gold_Medal == 1]
        gold = gold_medals.groupby(['Event', 'Year'])['ID'].count()

        gold_medals['Event_Frequency'] = gold_medals.groupby(
            ['Event', 'Year'])['ID'].transform('count')
        gold = gold_medals[gold_medals.Event_Frequency > 1]
        team_events = gold["Event"].unique()
        return team_events

    def tally_medal(olympics_NOC):
        tally = olympics_NOC[(olympics_NOC.No_Medal != 1) &
                             (olympics_NOC.Season == "Summer")]
        medal_tally = tally[['NOC',
                             'Year',
                             'Sport',
                             'Event',
                             'Medal',
                             'Bronze_Medal',
                             'Silver_Medal',
                             'Gold_Medal']]

        medal_tally_1 = medal_tally.drop_duplicates(['Medal', 'Event', 'Year'])

        medal_tally_1['Tally_Overall'] = medal_tally_1['Bronze_Medal'].astype(
            int) + medal_tally_1['Silver_Medal'].astype(int) + medal_tally_1['Gold_Medal'].astype(int)
        return medal_tally_1

    def tally_medal_year(medal_tally_1):
        medal_tally_by_year = pd.pivot_table(
            medal_tally_1,
            index=[
                'NOC',
                'Year'],
            values=[
                'Bronze_Medal',
                'Silver_Medal',
                'Gold_Medal',
                'Tally_Overall'],
            aggfunc=np.sum)

        medal_tally_by_year.sort_values('Tally_Overall', ascending=False)
        return medal_tally_by_year

    def tally_medal_overall(medal_tally_1):
        medal_tally_overall = pd.pivot_table(
            medal_tally_1,
            index=['NOC'],
            values=[
                'Bronze_Medal',
                'Silver_Medal',
                'Gold_Medal',
                'Tally_Overall'],
            aggfunc=np.sum)
        medal_tally_overall = medal_tally_overall.sort_values(
            'Tally_Overall', ascending=False)
        return medal_tally_overall

    def olympic_host_noc(olympics_host, noc_country):
        olympic_host_noc = olympics_host.merge(
            noc_country,
            left_on='Country',
            right_on=u'region',
            how='left')
        return olympic_host_noc

    @jit
    def previous_year(tally):
        new_tally = np.empty(tally.shape)
        new_tally[0] = 0
        for i in range(1, new_tally.shape[0]):
            new_tally[i] = tally[i - 1]
        return new_tally

    @jit
    def next_year(tally):
        new_tally = np.empty(tally.shape)
        new_tally[tally.shape[0] - 1] = 0
        for i in range(0, new_tally.shape[0] - 1):
            new_tally[i] = tally[i + 1]
        return new_tally

    @jit
    def getroutedistance(host_latitude, host_longitude):
        latitude = list(host_latitude)
        longitude = list(host_longitude)
        distance = []
        distance.append(0)
        for i in range(0, len(latitude) - 1):
            source = LatLon(latitude[i], longitude[i])
            destination = LatLon(latitude[i + 1], longitude[i + 1])
            distance.append(m2km(source.distanceTo(destination)))
        return distance
