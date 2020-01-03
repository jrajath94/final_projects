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
    >>> read_data_df('unknown.csv')
    Traceback (most recent call last):
    ...
    FileNotFoundError: [Errno 2] No such file or directory: ...

    >>> read_data_df(unknown.csv)
    Traceback (most recent call last):
    ...
    NameError: name 'unknown' is not defined


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



world_gdp = read_data_df('world_gdp.csv', skiprows=3)
noc_country = read_data_df('noc_regions.csv')
world_population = read_data_df('world_pop.csv')
olympics = read_data_df('athlete_events.csv')
olympics_host = read_data_df('olm2.csv')
world_hdi = read_data_df('HDI.csv')

print(get_shape(olympics))
print(get_shape(world_gdp))
print(get_shape(noc_country))
print(get_shape(world_population))
print(get_shape(olympics_host))
print(get_shape(world_hdi))

olympics.name = "Olympics"
world_gdp.name = "World GDP"
noc_country.name = "Country Codes"
world_population.name = "World Population"
olympics_host.name = "Olympics Host Countries"
world_hdi.name = "World HDI data"

get_info(olympics)
get_info(world_gdp)
get_info(noc_country)
get_info(world_population)
get_info(olympics_host)
get_info(world_hdi)

get_missing_values(olympics)

olympics['Medal'].fillna('No_Medal', inplace=True)

olympics.dtypes
world_gdp.dtypes
noc_country.dtypes
world_population.dtypes
olympics_host.dtypes
world_hdi.dtypes

# "One Hot Encoding of Medals Column"

##### https://www.datacamp.com/community/tutorials/categorical-data

olympics["Bronze_Medal"] = np.where(
    olympics["Medal"].str.contains("Bronze"), 1, 0)
olympics["Silver_Medal"] = np.where(
    olympics["Medal"].str.contains("Silver"), 1, 0)
olympics["Gold_Medal"] = np.where(olympics["Medal"].str.contains("Gold"), 1, 0)
olympics["No_Medal"] = np.where(
    olympics["Medal"].str.contains("No_Medal"), 1, 0)


world = world_gdp.drop(['Indicator Name', 'Indicator Code'], axis=1)


olympics_NOC = olympics.merge(
    noc_country,
    left_on='NOC',
    right_on='NOC',
    how='left')

olympics_host['Year'].fillna(-999, inplace=True)
olympics_host['Winter'].fillna("No_Olympics", inplace=True)
olympics_host['Summer (Olympiad)'].fillna("No_Olympics", inplace=True)
olympic_years = list(olympics_host["Year"].unique())
olympic_years = [str(int(element)) for element in olympic_years]

world_population_columns = list(world_population.columns.values)[4:]
for item in world_population_columns:
    if item not in olympic_years:
        world_population.drop(item, axis=1, inplace=True)
olympics_NOC_population = olympics_NOC.merge(
    world_population,
    left_on='NOC',
    right_on='Country Code',
    how='left').reset_index(
        drop=True)


olympics_NOC_population


list(world_gdp.columns)

world_gdp_columns = list(world_gdp.columns.values)[4:]
for item in world_gdp_columns:
    if item not in olympic_years:
        world_gdp.drop(item, axis=1, inplace=True)
olympics_NOC_gdp = olympics_NOC.merge(
    world_population,
    left_on='NOC',
    right_on='Country Code',
    how='left').reset_index(
    drop=True)
olympics_NOC_gdp


olympics_NOC_population


# American Athletes vs World , Medals Tally in Olympics

american_olympians = olympics_NOC.loc[(olympics_NOC.NOC == 'USA') & (
    olympics_NOC.Medal == 'Gold') & (olympics_NOC.Season == 'Summer')]
american_olympians

american_olympians.shape

pd.options.display.max_rows = 999
# We can see that the Gold medals won by America is 1127 , but the number
# os rows returned is 1951 . THis is due to medallists in team events
# being counted as individual medallists. Eg : - Soccer has 11 players ,
# It will be counted as 11 medals instead of 1 .


# Let's Find out Team Event

gold_medals = olympics_NOC[olympics_NOC.Gold_Medal == 1]
gold = gold_medals.groupby(['Event', 'Year'])['ID'].count()
gold
gold_medals['Event_Frequency'] = gold_medals.groupby(['Event', 'Year'])[
    'ID'].transform('count')
gold = gold_medals[gold_medals.Event_Frequency > 1]
team_events = gold["Event"].unique()
team_events

# Medals Tally

tally = olympics_NOC[(olympics_NOC.No_Medal != 1) &
                     (olympics_NOC.Season == "Summer")]
medal_tally = tally[['NOC', 'Year', 'Sport', 'Event',
                     'Medal', 'Bronze_Medal', 'Silver_Medal', 'Gold_Medal']]

medal_tally_1 = medal_tally.drop_duplicates(['Medal', 'Event', 'Year'])
medal_tally_1

# Gold Medallists of USA -

medal_tally_1[(medal_tally_1['Medal'] == 'Gold')
              & (medal_tally_1['NOC'] == 'USA')]

# Silver Medallists of USA -


medal_tally_1[(medal_tally_1['Medal'] == 'Silver')
              & (medal_tally_1['NOC'] == 'USA')]

# Bronze Medal Tally of USA -

medal_tally_1[(medal_tally_1['Medal'] == 'Bronze')
              & (medal_tally_1['NOC'] == 'USA')]


medal_tally_1.columns


medal_tally_1['Tally_Overall'] = medal_tally_1['Bronze_Medal'].astype(
    int) + medal_tally_1['Silver_Medal'].astype(int) + medal_tally_1['Gold_Medal'].astype(int)
sns.set()
medal_tally_usa = medal_tally_1[medal_tally_1['NOC'] == 'USA']
medal_tally_usa = pd.pivot_table(
    medal_tally_usa,
    index=[
        'NOC',
        'Year'],
    values=[
        'Bronze_Medal',
        'Silver_Medal',
        'Gold_Medal',
        'Tally_Overall'],
    aggfunc=np.sum)
medal_tally_usa.plot()


# USA's Medal Tally by Year (Sorted by Medal Tally )

medal_tally_usa.sort_values('Tally_Overall', ascending=False)

# Medal Tally of All Countries By Year (Sorted by Overall Tally)

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

# Medal Tally of All Countries Overall (1896 - 2016) (Sorted by Overall Tally)

medal_tally_overall = pd.pivot_table(
    medal_tally_1,
    index=['NOC'],
    values=[
        'Bronze_Medal',
        'Silver_Medal',
        'Gold_Medal',
        'Tally_Overall'],
    aggfunc=np.sum)
m = medal_tally_overall.sort_values('Tally_Overall', ascending=False)
m


# Pie Chart of Medal Distribution of top 20 Countries

sns.set()

medal_tally_overall = pd.pivot_table(
    m.head(20),
    index=['NOC'],
    values=['Tally_Overall'],
    aggfunc=np.sum).plot(
        kind='pie',
        subplots=True,
        figsize=(
            50,
            100),
    autopct='%1.1f%%',
    textprops={
            'fontsize': 40})

# Home Advantage ?

olympic_host_noc = olympics_host.merge(
    noc_country,
    left_on='Country',
    right_on=u'region',
    how='left')

olympic_host_noc = olympic_host_noc.drop_duplicates(
    ['Country', 'Year', 'Summer (Olympiad)'])
olympic_host_noc = olympic_host_noc.drop_duplicates(
    ['Country', 'Year', 'Winter'])
olympic_host_noc = olympic_host_noc.drop_duplicates(
    ['Country', 'Year'], keep='last')
olympic_host_noc = olympic_host_noc[['Year', 'NOC']]
olympic_host_noc.columns = ['Year_of_Hosting', 'Host_Country_code']
olympic_host_noc

# In[42]:


medal_tally_by_year.reset_index(inplace=True)
medal_tally_by_year = medal_tally_by_year[['NOC', 'Year', 'Tally_Overall']]
total_medals = medal_tally_by_year[['NOC', 'Year', 'Tally_Overall']]

# https://stackoverflow.com/questions/34855859/is-there-a-way-in-pandas-to-use-previous-row-value-in-dataframe-apply-when-previ

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


total_medals['Previous_Year_Tally'] = previous_year(
    total_medals['Tally_Overall'])
total_medals['Next_Year_Tally'] = next_year(total_medals['Tally_Overall'])

host_country_tally = olympic_host_noc.merge(
    total_medals, left_on=[
        'Host_Country_code', 'Year_of_Hosting'], right_on=[
            'NOC', 'Year'], how='left')

host = host_country_tally.dropna()
host = host[['Year_of_Hosting',
             'Host_Country_code',
             'Previous_Year_Tally',
             'Tally_Overall',
             'Next_Year_Tally']]
host


olympics_NOC.columns


total_medals['Tally_Overall']

total_medals['Tally_Overall'].shape[0]


# total_medals['Previous_Year_Tally'] = total_medals['Tally_Overall'].shift(1)

total_medals

olympic_host_noc.columns

host_country_tally.columns

# Prediction

predictor = medal_tally_by_year

gdp = olympics_NOC_gdp.drop_duplicates(['Country', 'Year'])
gdp = gdp.drop(['ID',
                'Name',
                'Sex',
                'Age',
                'Height',
                'Weight',
                'Games',
                'Season',
                'City',
                'Sport',
                'Event',
                'Medal',
                'Bronze_Medal',
                'Silver_Medal',
                'Gold_Medal',
                'No_Medal',
                'region',
                'notes',
                'Country',
                'Country Code',
                'Indicator Name',
                'Indicator Code'],
               axis=1)
gdp.dropna(inplace=True)
gdp = gdp[gdp.Year >= 1960]
pivot_gdp = pd.melt(
    world,
    id_vars=[
        'Country Name',
        'Country Code'],
    var_name='Year',
    value_name='GDP')  # inspired
pivot_gdp.dropna(inplace=True)

pivot_gdp

pop = world_population.drop(world_population.columns[[2, 3]], axis=1)
pivot_pop = pd.melt(
    pop,
    id_vars=[
        'Country',
        'Country Code'],
    var_name='Year',
    value_name='Population')  # inspired
pivot_pop


gdp_pop = pivot_gdp.merge(
    pivot_pop, left_on=[
        'Country Code', 'Year'], right_on=[
            'Country Code', 'Year'], how='left')
gdp_pop.drop_duplicates(['Country Code', 'Year'])
gdp_pop.dropna(inplace=True)

gdp_pop = gdp_pop[['Country Name',
                   'Country Code', 'Year', 'GDP', 'Population']]
gdp_pop
predictor['Year'] = predictor['Year'].astype('str')
pred = gdp_pop.merge(
    predictor, left_on=[
        'Country Code', 'Year'], right_on=[
            'NOC', 'Year'], how='left')
predictor['Year'] = predictor['Year'].astype('str')
pred.dropna()
pred['Per_Capita_GDP'] = pred['GDP'] / pred['Population']
pred.dropna(inplace=True)
pred

correlation_matrix = pred.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')

scaler = preprocessing.MinMaxScaler()
A = pred[['GDP', 'Population', 'Per_Capita_GDP']]
A = scaler.fit_transform(A)
B = pred['Tally_Overall']

model = sm.OLS(B, A)
results = model.fit()
print(results.summary())

reg = RandomForestRegressor(n_estimators=100)
reg.fit(A, B)

print("Random Forest Regression : {:.2f}".format(reg.score(A, B)))
B_predict = reg.predict(A)

rmse = np.sqrt(metrics.mean_squared_error(B, B_predict))

rmse = np.sqrt(metrics.mean_squared_error(B, B_predict))
print("RMSE Of Random Forest Regressor " + str(rmse))
plt.scatter(B, B_predict)


world_hdi.dropna(inplace=True)
pivot_hdi = pd.melt(
    world_hdi,
    id_vars=['HDI'],
    value_name='Human_Dev_Index')  # inspired
pivot_hdi.dropna(inplace=True)
pivot_hdi_noc = pivot_hdi.merge(
    noc_country,
    left_on='HDI',
    right_on='region',
    how='left')

pivot_hdi_noc.drop_duplicates(['variable', 'region', 'HDI'], keep='last')

pivot_hdi_noc = pivot_hdi_noc[['NOC', 'variable', 'Human_Dev_Index', 'HDI']]
pivot_hdi_noc.dropna(inplace=True)
pivot_hdi_noc.columns = [
    'NOC_HDI',
    'Year_HDI',
    'Human_Dev_Index',
    'Country_HDI']
pivot_hdi_noc


noc_country.columns

pre = pred.merge(
    pivot_hdi_noc, left_on=[
        'Country Name', 'Year'], right_on=[
            'Country_HDI', 'Year_HDI'], how='left')
pre.dropna(inplace=True)

MinMax = preprocessing.MinMaxScaler()
A = pre[['GDP', 'Population', 'Per_Capita_GDP', 'Human_Dev_Index']]
A = MinMax.fit_transform(A)
B = pre['Tally_Overall']
reg = RandomForestRegressor(n_estimators=100)
reg.fit(A, B)
model = sm.OLS(B, A)
results = model.fit()
print(results.summary())
print("\n \nRMSE Of Linear Regressor " + str(rmse))

print("\n \nRandom Forest Regression Fit: {:.2f}".format(reg.score(A, B)))
B_predict = reg.predict(A)

rmse = np.sqrt(metrics.mean_squared_error(B, B_predict))

print("\n \nRMSE Of Random Forest Regressor " + str(rmse))
plt.scatter(B, B_predict)
plt.title("Scatter Plot for Random Forrest Regressor")


pre.shape


pred.dtypes


correlation_matrix = pre.corr()
correlation_matrix.style.background_gradient(cmap='coolwarm')


# interesting Stat -

# how far were the cities that hosted Olympics Each Year?

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


olympics_host['distance(KM)'] = getroutedistance(
    olympics_host['latitude'], olympics_host['longitude'])

olympics_host


historical = olympics_NOC.dropna(subset=['Height', 'Weight']).reset_index()
history_height_mean = historical.groupby(['Year'])['Height'].mean()
history_height_mean.plot()

history_height_std = historical.groupby(['Year'])['Height'].std()
history_height_std.plot()

history_weight_mean = historical.groupby(['Year'])['Weight'].mean()
history_weight_mean.plot()


history_weight_std = historical.groupby(['Year'])['Weight'].std()
history_weight_std.plot()

# We Analyse the Claims made by a Research Paper based on 1972 Olympics , when it comes to the Height , Weight and Age of the athletes
# Research Paper - "Standards on age, height and weight in Olympic running events for men"
# The paper claims that Medalists in 100m Sprint event are Taller and heavier than other participants.
# Additionally, we try to explore this claim for Age and BMI as well.
# Overall Participants Average Height ,Height of the Gold Medalist and
# Height of All Medalists ( Gold , Silver , Bronze ) of 100m Running Event
# From 1896 -2016

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
    sprinters_gold
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

sprinter_stats(olympics_NOC, 'Height')

sprinter_stats(olympics_NOC, 'Weight')

sprinter_stats(olympics_NOC, 'Age')
