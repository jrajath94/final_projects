#!/usr/bin/env python
# coding: utf-8

import warnings

import dask.dataframe as ddf
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings('ignore')


def read_data(file, **kwargs):
    """
    This function is used to read csv files

    >>> read_data()
    Traceback (most recent call last):
    ...
    TypeError: read_data() missing 1 required positional argument: 'file'
    >>> read_data(abcd.csv)
    Traceback (most recent call last):
    ...
    NameError: name 'abcd' is not defined
    >>> read_data('Tests/olm2_test.csv')
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
    This function is used to read csv files using Dask

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
    >>> read_data_df('Tests/olm2_test.csv' , columns = 'asd')
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
    Get the shape of the Dataframes

    >>> get_shape()
    Traceback (most recent call last):
    ...
    TypeError: get_shape() missing 1 required positional argument: 'dataframe'


    >>> get_shape('unknown.dataframe')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'shape'
    >>> world_hdi_test = read_data_df('Data/hdi.csv')
    >>> olympics_host_test = read_data_df('Data/olm2.csv')
    >>> get_shape(world_hdi_test)
    (270, 10)
    >>> get_shape(olympics_host_test)
    (51, 7)

    :param dataframe: Pandas Dataframe
    :return: Shape of the Dataframe
    """
    return dataframe.shape


def get_stats(dataframe):
    """
    This function is used to get the stats from a dataframe

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
    >>> noc_country_test = read_data_df('Data/noc_regions.csv')
    >>> world_population_test = read_data_df('Data/world_pop.csv')

    >>> get_stats(noc_country_test)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 230 entries, 0 to 229
    Data columns (total 3 columns):
    NOC       230 non-null object
    region    227 non-null object
    notes     21 non-null object
    dtypes: object(3)
    memory usage: 5.5+ KB

    """
    return dataframe.info()


def get_missing_values(dataframe):
    """
    This function is used to get the missing values count from a dataframe

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

    >>> noc_country_test = read_data_df('Data/noc_regions.csv')
    >>> get_missing_values(noc_country_test)
    NOC         0
    region      3
    notes     209
    dtype: int64

    """
    return dataframe.isnull().sum()


def get_info(dataframe):
    """
    This function is used to get the rows and columns from a dataframe


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

    This function is used to get the stats of 100 M sprinters from the Olympics Dataframe

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


def FemaleTrend(olympics_NOC, Sex, Season):
    """
    In this function, the rise in the female trend is analyzed over the summer and the winter season

    >>> FemaleTrend('unknown.dataframe', 'Sex')
    Traceback (most recent call last):
    ...
    TypeError: FemaleTrend() missing 1 required positional argument: 'Season'

    >>> FemaleTrend(df, unknown_argument)
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined


    :param olympics_NOC:
    :return:
    """
    Femalecount = olympics_NOC[(olympics_NOC['Sex'] == Sex) & (olympics_NOC['Season'] == Season)].groupby(
        ['Year'])['Name'].count().reset_index(name="Female Trend")
    print(Femalecount)
    Femalecount.plot(kind='line', x='Year', y='Female Trend')


def events(olympics_NOC, Season):
    """
    In this function, the count of all different events is displayed in bar graph over the Summer and Winter Olympics

    >>> events_summer('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    NameError: name 'events_summer' is not defined

    >>> events_summer(df, unknown_argument)
    Traceback (most recent call last):
    ...
    NameError: name 'events_summer' is not defined


    :param olympics_NOC_gdp:
    :return:
    """
    # Plotting Event Count over the years in Summer Olympic

    Eventcount = olympics_NOC[olympics_NOC['Season'] == Season].groupby(
        ['Year'])['Event'].nunique().reset_index(name="Eventcount")
    print(Eventcount)
    fig, ax = plt.subplots(figsize=(28, 6))
    Eventcount.plot(kind='bar', x='Year', y='Eventcount', ax=ax, color='red')

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2.,
            p.get_height(),
            '%d' % int(
                p.get_height()),
            fontsize=12,
            color='black',
            ha='center',
            va='bottom')


def athletes(olympics_NOC, Season):
    """
    In this function, the count of all athlete participation is displayed in bar graph over the Summer and Winter Olympics

    >>> athletes()
    Traceback (most recent call last):
    ...
    TypeError: athletes() missing 2 required positional arguments: 'olympics_NOC' and 'Season'

    >>> athletes('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: athletes() missing 1 required positional argument: 'Season'

    >>> athletes(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined


    :param olympics_NOC:
    :return:
    """

    # Plotting Athlete count over the years in Summer Olympic

    Athletecount = olympics_NOC[olympics_NOC['Season'] == Season].groupby(
        ['Year'])['Name'].nunique().reset_index(name="Athletecount")
    print(Athletecount)
    fig, ax = plt.subplots(figsize=(22, 6))
    Athletecount.plot(
        kind='bar',
        x='Year',
        y='Athletecount',
        ax=ax,
        color='red')

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2.,
            p.get_height(),
            '%d' % int(
                p.get_height()),
            fontsize=12,
            color='black',
            ha='center',
            va='bottom')
