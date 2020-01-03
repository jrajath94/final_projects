#!/usr/bin/env python
# coding: utf-8

from pygeodesy.utily import m2km
from pygeodesy.ellipsoidalVincenty import LatLon
from numba import jit
import dask.dataframe as ddf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def read_data(file, **kwargs):
    """
    >>> read_data()
    Traceback (most recent call last):
    ...
    TypeError: read_data() missing 1 required positional argument: 'file'
    >>> read_data(abcd.csv)
    Traceback (most recent call last):
    ...
    NameError: name 'abcd' is not defined
    >>> read_data('abcd.csv')
    Traceback (most recent call last):
    ...
    FileNotFoundError: [Errno 2] File b'abcd.csv' does not exist: b'abcd.csv'

    >>> read_data('athlete_events.csv' , columns = 'asd')
    Traceback (most recent call last):
    ...
    TypeError: parser_f() got an unexpected keyword argument 'columns'


    :param file: CSV Files
    :param kwargs:
    :return: Pandas Dataframe
    """
    filename = pd.read_csv(file, **kwargs)
    return filename


def read_data_df(file, **kwargs):
    """
    >>> read_data_df()
    Traceback (most recent call last):
    ...
    TypeError: read_data_df() missing 1 required positional argument: 'file'

    >>> read_data_df('Tests/olm2_test.csv')
             Host City    Country Summer (Olympiad)  ...  Year   latitude   longitude
    0      Albertville     France               NaN  ...  1992  45.675500    6.392700
    1        Barcelona      Spain               XXV  ...  1992  41.388787    2.158985
    2      Lillehammer     Norway               NaN  ...  1994  61.115300   10.466200
    3          Atlanta        USA              XXVI  ...  1996  33.748995  -84.387985
    4           Nagano      Japan               NaN  ...  1998  36.650000  138.183334
    5           Sydney  Australia             XXVII  ...  2000 -33.867850  151.207321
    6   Salt Lake City        USA               NaN  ...  2002  40.758701 -111.876183
    7           Athens     Greece            XXVIII  ...  2004  37.983333   23.733334
    8            Turin      Italy               NaN  ...  2006  45.070300    7.686900
    9          Beijing      China              XXIX  ...  2008  39.907498  116.397224
    10       Vancouver     Canada               NaN  ...  2010  49.249657 -123.119339
    11          London         UK               XXX  ...  2012  51.508415   -0.125533
    12           Sochi     Russia               NaN  ...  2014  43.600000   39.730278
    13  Rio de Janeiro     Brazil              XXXI  ...  2016 -22.902778  -43.207501
    <BLANKLINE>
    [14 rows x 7 columns]
    >>> read_data_df('athlete_events.csv' , columns = 'asd')
    Traceback (most recent call last):
    ...
    TypeError: parser_f() got an unexpected keyword argument 'columns'



    :param file: Input CSV File
    :param kwargs: addtional arguments
    :return: Pandas Dataframe
    """
    # filename = pd.read_csv(file, **kwargs)
    df = ddf.read_csv(file, **kwargs, blocksize=1000000)
    df = df.compute(scheduler='processes')
    return df


def get_shape(dataframe):
    """
    >>> get_shape()
    Traceback (most recent call last):
    ...
    TypeError: get_shape() missing 1 required positional argument: 'dataframe'


    >>> get_shape('unknown.dataframe')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'shape'

    >>> get_shape(Pandasdataframe)
    Traceback (most recent call last):
    ...
    NameError: name 'Pandasdataframe' is not defined

    :param dataframe: Pandas Dataframe
    :return: Shape of the Dataframe
    """
    return dataframe.shape


def get_stats(dataframe):
    """
    >>> get_stats()
    Traceback (most recent call last):
    ...
    TypeError: get_stats() missing 1 required positional argument: 'dataframe'


    >>> get_stats('unknown.dataframe')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'info'

    >>> get_stats(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    :param dataframe: Pandas Dataframe
    :return: Shape of the Dataframe

    :param dataframe: Pandas Dataframe
    :return: Information about the dataframe
    """
    return dataframe.info()


def get_missing_values(dataframe):
    """
    >>> get_missing_values()
    Traceback (most recent call last):
    ...
    TypeError: get_missing_values() missing 1 required positional argument: 'dataframe'


    >>> get_missing_values('unknown.dataframe')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'isnull'

    >>> get_missing_values(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    >>> get_missing_values(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    :param dataframe: Pandas Dataframe
    :return: Details of the missing values in the dataframe
    """
    return dataframe.isnull().sum()


def get_info(dataframe):
    """

    >>> get_info()
    Traceback (most recent call last):
    ...
    TypeError: get_info() missing 1 required positional argument: 'dataframe'


    >>> get_info('unknown.dataframe')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'columns'

    >>> get_info(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    >>> get_info(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined


   :param dataframe: Pandas Dataframe
   :return: Details of the Dataframe
   """
    columns = list(filter(None, list(dataframe.columns.values)))

    names = str(columns)[1:-1]
    row, col = dataframe.shape
    print(
        "\nThe Dataframe \"{}\" has {} rows and {} columns".format(
            dataframe.name,
            row,
            col))
    print("Column Names - {} \n".format(names))





def sprinter_stats(olympics, paramater):
    """
    >>> sprinter_stats()
    Traceback (most recent call last):
    ...
    TypeError: sprinter_stats() missing 2 required positional arguments: 'olympics' and 'paramater'

    >>> sprinter_stats('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: sprinter_stats() missing 1 required positional argument: 'paramater'

    >>> sprinter_stats(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> sprinter_stats('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: sprinter_stats() takes 2 positional arguments but 3 were given

    >>> sprinter_stats(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param summer_olympics:
    :return:
    """

    sprinters = olympics[(olympics['Event'] == "Athletics Men's 100 metres")]
    sprinters_height = sprinters.groupby(['Year'])[paramater].mean()

    sprinters_gold = olympics[(olympics['Medal'] == 'Gold') & (
        olympics['Event'] == "Athletics Men's 100 metres")]

    sprinters_gold_height = sprinters_gold.groupby(['Year'])[paramater].mean()

    med = ['Gold', 'Silver', 'Bronze']
    sprinters_medal = olympics[(olympics['Medal'].isin(med)) & (
        olympics['Event'] == "Athletics Men's 100 metres")]

    sprinters_medal_Height = sprinters_medal.groupby(['Year'])[
        paramater].mean()

    sprint = pd.merge(
        sprinters_height,
        sprinters_gold_height,
        on=['Year']).reset_index()

    sprint = pd.merge(sprint, sprinters_medal_Height, on=['Year'])

    sprint.columns = [
        'Year',
        'Overall Participants Average',
        'Gold Medalists',
        'All Medalists']
    # Athletics Men's 100 metres
    return sprint

