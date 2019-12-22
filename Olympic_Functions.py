#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from sklearn import metrics
import pandas as pd
import dask.dataframe as ddf
import dask.multiprocessing
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

    >>> read_data('olym.csv')
    Traceback (most recent call last):
    ...
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa0 in position 0: invalid start byte
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
    FileNotFoundError: [Errno 2] No such file or directory: '/Users/rj/PycharmProjects/final_projects/unknown.csv'

    >>> read_data_df(unknown.csv)
    Traceback (most recent call last):
    ...
    NameError: name 'unknown' is not defined

    >>> read_data_df('olym.csv')
    Traceback (most recent call last):
    ...
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa0 in position 0: invalid start byte
    >>> read_data_df('athlete_events.csv' , columns = 'asd')
    Traceback (most recent call last):
    ...
    TypeError: parser_f() got an unexpected keyword argument 'columns'



    :param file: Input CSV File
    :param kwargs: addtional arguments
    :return: Pandas Dataframe
    """
   # filename = pd.read_csv(file, **kwargs)
    df = ddf.read_csv(file,**kwargs, blocksize=1000000)
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
   # list(filter(None, test_list))
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
    columns =  list(filter(None,list(dataframe.columns.values)))
    #columns = [i for i in columns if i]
    names = str(columns)[1:-1]
    row, col = dataframe.shape
    print("\nThe Dataframe has {} rows and {} columns".format(row,col))
    
    print("Column Names - {} \n".format(names))

    


# In[2]:


def data_cleaning(olympics, noc_country,world_gdp , world_population, olympics_host):

    """

    >>> data_cleaning()
    Traceback (most recent call last):
    ...
    TypeError: data_cleaning() missing 5 required positional arguments: 'olympics', 'noc_country', 'world_gdp', 'world_population', and 'olympics_host'


    >>> data_cleaning('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: data_cleaning() missing 4 required positional arguments: 'noc_country', 'world_gdp', 'world_population', and 'olympics_host'
    >>> data_cleaning(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> data_cleaning('unknown.dataframe', 'country', 'gdp', 'population','host')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> data_cleaning(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined





    :param olympics: Olympics Dataset
    :param noc_country:  NOC codes of the country
    :param world_gdp: World GDP
    :param world_population: World Population by country
    :param olympics_host: Host city of the Olympics
    :return: Intermediate dafaframes that could be used in other parts of the code.
    """

    olympics['Medal'].fillna('DNW', inplace = True)

    #noc_country.drop('notes', axis = 1 , inplace = True)
    noc_country.rename(columns = {'region':'Country'}, inplace = True)

    #Ensuring Only 
    Merge_olympics = olympics.merge(noc_country,
                                    left_on = 'NOC',
                                    right_on = 'NOC',
                                    how = 'left')
    Merge_olympics.drop('notes', axis = 1 , inplace = True)


    world_gdp.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

    # The columns are the years for which the GDP has been recorded. This needs to brought into a single column for efficient
    # merging.
    world_gdp = pd.melt(world_gdp, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'GDP')

    # convert the year column to numeric
    world_gdp['Year'] = pd.to_numeric(world_gdp['Year'])

    merge_olympics_countrycode = Merge_olympics.merge(world_gdp[['Country Name', 'Country Code']].drop_duplicates(),
                                                left_on = 'Team',
                                                right_on = 'Country Name',
                                                how = 'left')

    merge_olympics_countrycode.drop('Country Name', axis = 1, inplace = True)

    # Merge to get gdp too
    olympics_merge_gdp = merge_olympics_countrycode.merge(world_gdp,
                                                    left_on = ['Country Code', 'Year'],
                                                    right_on = ['Country Code', 'Year'],
                                                    how = 'left')

    olympics_merge_gdp.drop('Country Name', axis = 1, inplace = True)


    world_population.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)

    world_population = pd.melt(world_population, id_vars = ['Country', 'Country Code'], var_name = 'Year', value_name = 'Population')

    # Change the Year to integer type
    world_population['Year'] = pd.to_numeric(world_population['Year'])


    olympics_new = olympics_merge_gdp.merge(world_population,
                                                left_on = ['Country Code', 'Year'],
                                                right_on= ['Country Code', 'Year'],
                                                how = 'left')

    olympics_new.drop('Country_y', axis = 1, inplace = True)
    print("Data Cleaning has been Completed")
    
    return Merge_olympics,merge_olympics_countrycode,olympics_merge_gdp,olympics_new,world_gdp,world_population,olympics_new 


# In[ ]:



# In[20]:


def correlation( Merge_olympics,merge_olympics_countrycode,olympics_merge_gdp,olympics_new):
    """
    >>> correlation()
    Traceback (most recent call last):
    ...
    TypeError: correlation() missing 4 required positional arguments: 'Merge_olympics', 'merge_olympics_countrycode', 'olympics_merge_gdp', and 'olympics_new'

    >>> correlation('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: correlation() missing 3 required positional arguments: 'merge_olympics_countrycode', 'olympics_merge_gdp', and 'olympics_new'
    >>> correlation(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> correlation('unknown.dataframe', 'country', 'gdp', 'population','host')
    Traceback (most recent call last):
    ...
    TypeError: correlation() takes 4 positional arguments but 5 were given
    >>> correlation('olympics','countrycode','gdp','olympics_new','misc')
    Traceback (most recent call last):
    ...
    TypeError: correlation() takes 4 positional arguments but 5 were given


    >>> correlation(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined


    :param Merge_olympics:
    :param merge_olympics_countrycode:
    :param olympics_merge_gdp:
    :param olympics_new:
    :return:
    """

    get_ipython().run_line_magic('pylab', 'inline')
    olympics_merge = olympics_new.loc[(olympics_new['Year'] > 1960) & (olympics_new['Season'] == "Summer"), :]

    # Reset row indices
    olympics_merge = olympics_merge.reset_index()
    olympics_host = pd.read_csv('olym.csv',encoding="ISO-8859–1")
    mergedframe = pd.merge(olympics_merge, olympics_host, left_on='City', right_on='Host City', how ='inner')

    #import numpy as np
    mergedframe['Medal'].fillna('DNW', inplace = True)
    mergedframe['Medal_Won'] = np.where(mergedframe.loc[:,'Medal'] == 'DNW', 0, 1)
    team_events = pd.pivot_table(mergedframe,
                                          index = ['Team', 'Year_x', 'Event'],
                                          columns = 'Medal',
                                          values = 'Medal_Won',
                                          aggfunc = 'sum',
                                         fill_value = 0).drop('DNW', axis = 1).reset_index()



    team_events = team_events.loc[team_events['Gold'] > 1, :]
    team_sports = team_events['Event'].unique()

    team_var = mergedframe['Event'].map(lambda x: x in team_sports)
    singleEvent = [not i for i in team_var]



    # rows where medal_won is 1
    medal_mask = mergedframe['Medal_Won'] == 1

    mergedframe['Team_Event'] = np.where(team_var & medal_mask, 1, 0)

    # Put 1 under singles event if medal is won and event not in team event list
    mergedframe['Single_Event'] = np.where(singleEvent & medal_mask, 1, 0)

    # Add an identifier for team/single event
    mergedframe['Event_Category'] = mergedframe['Single_Event'] + mergedframe['Team_Event']
    temp_medaltally = mergedframe.groupby(['Year_x', 'Team', 'Event', 'Medal'])[['Medal_Won', 'Event_Category']].agg('sum').reset_index()
    temp_medaltally['Medal_Won_Corrected'] = temp_medaltally['Medal_Won']/temp_medaltally['Event_Category']

    # print(medal_tally_agnostic)
    medalTotal = temp_medaltally.groupby(['Year_x','Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

    year_team_gdp = olympics_merge.loc[:, ['Year', 'Team', 'GDP']].drop_duplicates()

    medalTotal_gdp = medalTotal.merge(year_team_gdp,
                                       left_on = ['Year_x', 'Team'],
                                       right_on = ['Year', 'Team'],
                                       how = 'left')
    
    row5 = medalTotal_gdp['Medal_Won_Corrected'] > 0
    
    correlation = medalTotal_gdp.loc[row5, ['GDP', 'Medal_Won_Corrected']].corr()['Medal_Won_Corrected'][0]

    plt.plot(medalTotal_gdp.loc[row5, 'GDP'],
         medalTotal_gdp.loc[row5, 'Medal_Won_Corrected'] , 
         linestyle = 'none', 
         marker = 'x',
        alpha = 0.4 , color='red')
    plt.xlabel('GDP of Country')

    plt.ylabel('Count of Medals')
    plt.text(np.nanpercentile(medalTotal_gdp['GDP'], 99.6),
         max(medalTotal_gdp['Medal_Won_Corrected']) - 50,
         "Correlation = " + str(correlation))
    
    return mergedframe,medalTotal_gdp


# In[4]:


def team_performance(mergedframe, countries):
    """
    >>> team_performance()
    Traceback (most recent call last):
    ...
    TypeError: team_performance() missing 2 required positional arguments: 'mergedframe' and 'countries'

    >>> team_performance('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: team_performance() missing 1 required positional argument: 'countries'
    >>> team_performance(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> team_performance('unknown.dataframe', 'country', 'gdp', 'population','host')
    Traceback (most recent call last):
    ...
    TypeError: team_performance() takes 2 positional arguments but 5 were given

    >>> team_performance('olympics','countrycode')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'groupby'


    >>> team_performance(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param mergedframe:
    :param countries:
    :return:
    """
    medal_tally_agnostic = mergedframe.groupby(['Year_x', 'Team', 'Event', 'Medal'])[['Medal_Won', 'Event_Category']].agg('sum').reset_index()
    medal_tally_agnostic['Medal_Won_Corrected'] = medal_tally_agnostic['Medal_Won']/medal_tally_agnostic['Event_Category']
    
    row_mask_2 = medal_tally_agnostic['Team'].map(lambda x: x in countries)


    best_team_sports = pd.pivot_table(medal_tally_agnostic[row_mask_2],
                                      index = ['Team', 'Event'],
                                      columns = 'Medal',
                                      values = 'Medal_Won_Corrected',
                                      aggfunc = 'sum',
                                      fill_value = 0).sort_values(['Team', 'Gold'], ascending = [True, False]).reset_index()

    best_team_sports.drop(['Bronze', 'Silver', 'DNW'], axis = 1, inplace = True)
    best_team_sports.columns = ['Team', 'Event', 'Gold_Medal_Count']
    pd.set_option('display.max_columns', None) 
    pd.options.display.max_rows = 999
    
    best = best_team_sports.groupby('Team')
    return best,medal_tally_agnostic
    


# In[5]:


def home_advantage(medal_tally_agnostic,mergedframe):

    """
    >>> home_advantage()
    Traceback (most recent call last):
    ...
    TypeError: home_advantage() missing 2 required positional arguments: 'medal_tally_agnostic' and 'mergedframe'

    >>> home_advantage('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: home_advantage() missing 1 required positional argument: 'mergedframe'

    >>> home_advantage(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> home_advantage('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: home_advantage() takes 2 positional arguments but 3 were given

    >>> home_advantage(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined



    :param medal_tally_agnostic: Medal Tally of a country
    :return: Team medals by year and also gives the Home advantage table
    """

    medal_tally_agnostic['Medal_Won_Corrected'] = medal_tally_agnostic['Medal_Won']/medal_tally_agnostic['Event_Category']
    # print(medal_tally_agnostic)
    medal_tally = medal_tally_agnostic.groupby(['Year_x','Team'])['Medal_Won_Corrected'].agg('sum').reset_index()
    # x=mergedframe.drop(['ID','Sex','Age','Height','Weight','Year_y','Games','Event'],axis=1)
    # pd.set_option('display.max_columns', None) 
    # pd.options.display.max_rows = 999
    # x.head(100)
    y=mergedframe.drop(['ID','Sex','Age','Height','Weight','Year_y','Event'],axis=1)
    y.rename(columns = {'Year_x':'Year'}, inplace = True)
    medal_tally.head()
    # y.head(200)
    # z=y.groupby(['Season','Team'])['Name'].count()
    # pd.set_option('display.max_columns', None) 
    # pd.options.display.max_rows = 999
    # z.head(200)
    year_host_team = y[['Year', 'Country', 'Team','Season']].drop_duplicates()
    #print(year_host_team)


    #print(year_host_team)
    row_mask_4 = (year_host_team['Country'].str.strip() == year_host_team['Team'].str.strip())

    # add years in the year_host_team to capture one previous and one later year
    year_host_team['Prev_Year'] = year_host_team['Year'] - 4
    year_host_team['Next_Year'] = year_host_team['Year'] + 4

    # Subset only where host nation and team were the same
    year_host_team = year_host_team[row_mask_4]

    #print("asdf")
    #year_host_team.columns
    #medal_tally.columns

    # Calculate the medals won in each year where a team played at home. merge year_host_team with medal_tally on year and team
    year_host_team_medal = year_host_team.merge(medal_tally,
                                               left_on = ['Year', 'Team'],
                                               right_on = ['Year_x', 'Team'],
                                               how = 'left')
    #print(medal_tally)
    #print("asdf")
    #print(year_host_team)

    #print(year_host_team_medal)
    year_host_team_medal.rename(columns = {'Medal_Won_Corrected' : 'Medal_Won_Host_Year'}, inplace = True)

    # Calculate medals won by team in previous year
    year_host_team_medal = year_host_team_medal.merge(medal_tally,
                                                     left_on = ['Prev_Year', 'Team'],
                                                     right_on = ['Year_x', 'Team'],
                                                     how = 'left')
    #print(year_host_team_medal)
    year_host_team_medal.drop('Year_x_y', axis = 1, inplace = True)
    year_host_team_medal.rename(columns = {'Medal_Won_Corrected': 'Medal_Won_Prev_Year',
                                        'Year':'Year'}, inplace = True)

    # # Calculate the medals won by the team the year after they hosted.
    year_host_team_medal = year_host_team_medal.merge(medal_tally,
                                                     left_on = ['Next_Year', 'Team'],
                                                     right_on = ['Year_x', 'Team'],
                                                     how = 'left')
    #print(year_host_team_medal)
    year_host_team_medal.drop('Year_x', axis = 1, inplace = True)
    year_host_team_medal.rename(columns = {'Year': 'Year',
                                           'Medal_Won_Corrected' : 'Medal_Won_Next_Year'}, inplace = True)

    # # General formatting changes
    year_host_team_medal.drop(['Prev_Year', 'Next_Year'], axis = 1, inplace = True)
    year_host_team_medal.sort_values('Year', ascending = True, inplace = True)
    year_host_team_medal.reset_index(inplace = True, drop = True)


    # # column re-ordering
    year_host_team_medal = year_host_team_medal.loc[:, ['Year', 'Country', 'Team', 'Season','Medal_Won_Prev_Year', 'Medal_Won_Host_Year', 'Medal_Won_Next_Year']]

    year_host_team_medal.fillna(" - " , inplace = True  )
    
    return year_host_team_medal


# In[6]:


def top_10_countries_summer(noc_country,olympics):

    """

    >>> top_10_countries_summer()
    Traceback (most recent call last):
    ...
    TypeError: top_10_countries_summer() missing 2 required positional arguments: 'noc_country' and 'olympics'

    >>> top_10_countries_summer('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: top_10_countries_summer() missing 1 required positional argument: 'olympics'

    >>> top_10_countries_summer(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> top_10_countries_summer('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: top_10_countries_summer() takes 2 positional arguments but 3 were given

    >>> top_10_countries_summer('country','host')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers


    >>> top_10_countries_summer(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined


    :param noc_country: Country Codes
    :param olympics:  Olympic athletes dataset
    :return:
    """

    noc_country['Country'].fillna(noc_country['notes'], inplace=True)
    olympics['Medal'] = olympics['Medal'].fillna('No Medal')
    player = olympics.merge(noc_country, how='left', on='NOC')
    #print(athlete)
    top10_summer = player[(player['Season']=='Summer') & (player['Medal']!='No Medal')].groupby('Country').count().reset_index()[['Country','Medal']].sort_values('Medal', ascending=False).head(10)
    f, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Country", y="Medal", data=top10_summer, label="Country", color="red")

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2,p.get_height(),

                '{:1.0f}'.format(p.get_height()),
                ha="center")

    #ax.set_xlabel('Country', size=14, color="orange")
    #ax.set_ylabel('Total Medals Won', size=14, color="green")
    ax.set_title('Top 10 countries with total medals in Summer Olympic games', size=16)
    plt.show()
    return player


# In[7]:


def top_10_countries_winter(player):
    """
    >>> top_10_countries_winter()
    Traceback (most recent call last):
    ...
    TypeError: top_10_countries_winter() missing 1 required positional argument: 'player'

    >>> top_10_countries_winter('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers


    >>> top_10_countries_winter(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined


    >>> top_10_countries_winter('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: top_10_countries_winter() takes 1 positional argument but 3 were given

    >>> top_10_countries_winter('country')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> top_10_countries_winter(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined


    :param player:
    :return:
    """
    top10_winter = player[(player['Season']=='Winter') & (player['Medal']!='No Medal')].groupby('Country').count().reset_index()[['Country','Medal']].sort_values('Medal', ascending=False).head(10)
    f, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Country", y="Medal", data=top10_winter, label="Country", color="purple")

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2,p.get_height(),

                '{:1.0f}'.format(p.get_height()),
                ha="center")

    ax.set_title('Top 10 countries with total medals in Winter Olympic games', size=16)
    plt.show()


# In[8]:


def athletes_edition(olympics,edition):
    """

    >>> athletes_edition()
    Traceback (most recent call last):
    ...
    TypeError: athletes_edition() missing 2 required positional arguments: 'olympics' and 'edition'


    >>> athletes_edition('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: athletes_edition() missing 1 required positional argument: 'edition'

    >>> athletes_edition(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> athletes_edition('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: athletes_edition() takes 2 positional arguments but 3 were given

    >>> athletes_edition(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined



    :param olympics:
    :return:
    """
    Data = olympics[olympics['Season']==edition]

    Athletes = Data.pivot_table(Data, index=['Year'], aggfunc=lambda x: len(x.unique())).reset_index()[['Year','ID']]
    Sports = Data.groupby('Year')['Sport'].nunique().reset_index()
    Events = Data.groupby('Year')['Event'].nunique().reset_index()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(22,18))

    sns.barplot(x='Year', y='ID', data=Athletes, ax=ax[0], color="red")
    sns.barplot(x='Year', y='Sport', data=Sports, ax=ax[1], color="blue")
    sns.barplot(x='Year', y='Event', data=Events, ax=ax[2], color="orange")

    j = 0
    for i in ['Athletes', 'Sports', 'Events']:
        ax[j].set_xlabel('Year', size=14)
        ax[j].set_ylabel(i, size=14)
        ax[j].set_title(i + ' in '+edition+'  Olympics ', size=18)
        j = j + 1

    for i in range(3):
        for p in ax[i].patches:
            ax[i].text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),
                    fontsize=12, color='black', ha='center', va='bottom')
    plt.show()


# In[9]:





# In[10]:


def BMI_by_event_participants(olympics ,gender):
    """
    >>> BMI_by_event_participants()
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_event_participants() missing 2 required positional arguments: 'olympics' and 'gender'

    >>> BMI_by_event_participants('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_event_participants() missing 1 required positional argument: 'gender'

    >>> BMI_by_event_participants(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> BMI_by_event_participants('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_event_participants() takes 2 positional arguments but 3 were given

    >>> BMI_by_event_participants('country','host')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers


    >>> BMI_by_event_participants(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param olympics:
    :param gender:
    :return:
    """
    olympics['BMI'] = olympics['Weight']/(olympics['Height']/100)**2
    olympics['BMI']
    Athletics_male=olympics[(olympics["Sex"]==gender) & (olympics["Sport"]=='Athletics')]   
    Swimmers_male=olympics[ (olympics["Sex"]==gender) & (olympics["Sport"]=='Swimming')] 
    Wrestlers_male=olympics[(olympics["Sex"]==gender) & (olympics["Sport"]=='Wrestling')]    
    Footballers_male=olympics[ (olympics["Sex"]==gender) & (olympics["Sport"]=='Football')]

    f,ax=plt.subplots(1,4,figsize=(15,7))

    Athletics_male.BMI.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='red')

    ax[0].set_title('BMI Distribution of Gold Athletics')

    x1=list(range(15,50,5))

    ax[0].set_xticks(x1)

    Swimmers_male.BMI.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='green')

    ax[1].set_title('BMI Distribution of Gold Swimming')

    x2=list(range(10,40,5))

    ax[1].set_xticks(x2)




    Wrestlers_male.BMI.plot.hist(ax=ax[2],bins=30,edgecolor='black',color='blue')

    ax[2].set_title('BMI Distribution of Gold Wrestling')

    x3=list(range(10,40,5))

    ax[2].set_xticks(x3)


    Footballers_male.BMI.plot.hist(ax=ax[3],bins=30,edgecolor='black',color='yellow')

    ax[3].set_title('BMI Distribution of Gold Football')

    x4=list(range(10,40,5))

    ax[3].set_xticks(x4)


    plt.show()


# In[ ]:



# In[11]:


def BMI_by_event_gold_medalists(olympics,gender):
    """
    >>> BMI_by_event_gold_medalists()
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_event_gold_medalists() missing 2 required positional arguments: 'olympics' and 'gender'

    >>> BMI_by_event_gold_medalists('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_event_gold_medalists() missing 1 required positional argument: 'gender'

    >>> BMI_by_event_gold_medalists(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> BMI_by_event_gold_medalists('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_event_gold_medalists() takes 2 positional arguments but 3 were given

    >>> BMI_by_event_gold_medalists('country','host')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers


    >>> BMI_by_event_gold_medalists(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param olympics:
    :param gender:
    :return:
    """
    Athletics=olympics[(olympics["Medal"]=='Gold') & (olympics["Sex"]==gender) & (olympics["Sport"]=='Athletics')].loc[:,["BMI","Sport","Medal"]]    
    Swimmers=olympics[(olympics["Medal"]=='Gold') & (olympics["Sex"]==gender) & (olympics["Sport"]=='Swimming')].loc[:,["BMI","Sport","Medal"]]    
    Wrestlers=olympics[(olympics["Medal"]=='Gold') & (olympics["Sex"]==gender)  & (olympics["Sport"]=='Wrestling')].loc[:,["BMI","Sport","Medal"]]    
    Footballers=olympics[(olympics["Medal"]=='Gold') & (olympics["Sex"]==gender) & (olympics["Sport"]=='Football')].loc[:,["BMI","Sport","Medal"]]    

    f,ax=plt.subplots(1,4,figsize=(15,7))

    Athletics.BMI.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='red')
    plt.title("Male")

    ax[0].set_title('BMI Distribution of Athletics (Gold)')

    x1=list(range(15,50,5))

    ax[0].set_xticks(x1)

    Swimmers.BMI.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='green')

    ax[1].set_title('BMI Distribution of Swimming (Gold)')

    x2=list(range(10,40,5))

    ax[1].set_xticks(x2)




    Wrestlers.BMI.plot.hist(ax=ax[2],bins=30,edgecolor='black',color='blue')

    ax[2].set_title('BMI Distribution of Wrestling (Gold)')

    x3=list(range(10,40,5))

    ax[2].set_xticks(x3)


    Footballers.BMI.plot.hist(ax=ax[3],bins=30,edgecolor='black',color='yellow')

    ax[3].set_title('BMI Distribution of Football (Gold)')

    x4=list(range(10,40,5))

    ax[3].set_xticks(x4)


    plt.show()



# In[ ]:



# In[12]:


# Evolution Based on BMI
def BMI_by_time_gold_medalists(olympics):

    """
    >>> BMI_by_time_gold_medalists()
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_time_gold_medalists() missing 1 required positional argument: 'olympics'

    >>> BMI_by_time_gold_medalists('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> BMI_by_time_gold_medalists(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> BMI_by_time_gold_medalists('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_time_gold_medalists() takes 1 positional argument but 3 were given


    >>> BMI_by_time_gold_medalists('country','host')
    Traceback (most recent call last):
    ...
    TypeError: BMI_by_time_gold_medalists() takes 1 positional argument but 2 were given



    >>> BMI_by_time_gold_medalists(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param olympics:
    :return:
    """
    years = [1896,1900,1904,1908,1912,1916,1920,1924,1928]

    Athletics_1896=olympics[(olympics["Medal"]=='Gold') & (olympics["Year"].isin(years)) & (olympics["Sport"]=='Athletics')]

    #Athletics_2016= olympics[(olympics["Year"]=='2016') & (olympics["Sport"]=='Athletics')].loc[:,["BMI","Sport","Medal"]]    
    Athletics_2016=olympics[(olympics["Medal"]=='Gold') & (olympics["Year"]==2016) & (olympics["Sport"]=='Athletics')]


    Wrestlers_1896=olympics[(olympics["Medal"]=='Gold') & (olympics["Year"].isin(years)) & (olympics["Sport"]=='Wrestling')]    
    Wrestlers_2016=olympics[(olympics["Medal"]=='Gold') & (olympics["Year"]==2016) & (olympics["Sport"]=='Wrestling')]

    f,ax=plt.subplots(1,4,figsize=(15,7))

    Athletics_1896.BMI.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='purple')

    ax[0].set_title('BMI Athletics Medals 1896-1952')

    x1=list(range(15,50,5))

    ax[0].set_xticks(x1)

    Athletics_2016.BMI.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='purple')

    ax[1].set_title('BMI Athletics Medals 2016 ')

    x2=list(range(10,40,5))

    ax[1].set_xticks(x2)


    Wrestlers_1896.BMI.plot.hist(ax=ax[2],bins=30,edgecolor='black',color='blue')

    ax[2].set_title('BMI Wrestling 1896-1952 ')

    x3=list(range(10,40,5))

    ax[2].set_xticks(x3)


    Wrestlers_2016.BMI.plot.hist(ax=ax[3],bins=30,edgecolor='black',color='blue')

    ax[3].set_title('BMI Wrestling 2016')

    x4=list(range(10,40,5))

    ax[3].set_xticks(x4)


    plt.show()


# In[13]:


def participants(olympics):
    """

    >>> participants()
    Traceback (most recent call last):
    ...
    TypeError: participants() missing 1 required positional argument: 'olympics'

    >>> participants('unknown.dataframe')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'Season'

    >>> participants(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> participants('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: participants() takes 1 positional argument but 3 were given

    >>> participants(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined



    :param olympics:
    :return:
    """
    summer_olympics = olympics[(olympics.Season == 'Summer')]
#    summer_olympics.groupby(['Year'])['ID'].count().reset_index(drop=True)
    return summer_olympics


# In[14]:


def sprinter_Height(summer_olympics):
    """
    >>> sprinter_Height()
    Traceback (most recent call last):
    ...
    TypeError: sprinter_Height() missing 1 required positional argument: 'summer_olympics'

    >>> sprinter_Height('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> sprinter_Height(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> sprinter_Height('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: sprinter_Height() takes 1 positional argument but 3 were given

    >>> sprinter_Height(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param summer_olympics:
    :return:
    """

    sprinters = summer_olympics[summer_olympics['Event']=="Athletics Men's 100 metres"]
    sprinters_height  = sprinters.groupby(['Year'])['Height'].mean()

    sprinters_gold = summer_olympics[(summer_olympics['Medal']=='Gold') & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    sprinters_gold
    sprinters_gold_height = sprinters_gold.groupby(['Year'])['Height'].mean()

    #sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin('Gold','Silver','Bronze')
                                          #     & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    med=['Gold','Silver','Bronze']
    sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin(med)) & (summer_olympics['Event']=="Athletics Men's 100 metres")]

    sprinters_medal
    sprinters_medal_Height = sprinters_medal.groupby(['Year'])['Height'].mean()

    sprint = pd.merge(sprinters_height, sprinters_gold_height, on=['Year']).reset_index()

    sprint = pd.merge(sprint, sprinters_medal_Height, on=['Year'])

    sprint.columns = ['Year', 'Overall Participants Average', 'Gold Medalists', 'All Medalists']
    #Athletics Men's 100 metres
    return sprint

#sprint.style.applymap(lambda x: 'background-color : yellow' if x>sprint.iloc[3,3] else '')


# In[15]:


def sprinter_weight(summer_olympics):
    """
    >>> sprinter_weight()
    Traceback (most recent call last):
    ...
    TypeError: sprinter_weight() missing 1 required positional argument: 'summer_olympics'

    >>> sprinter_weight('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> sprinter_weight(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> sprinter_weight('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: sprinter_weight() takes 1 positional argument but 3 were given

    >>> sprinter_weight(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined



    :param summer_olympics:
    :return:
    """
    sprinters = summer_olympics[summer_olympics['Event']=="Athletics Men's 100 metres"]
    sprinters_Weight  = sprinters.groupby(['Year'])['Weight'].mean()

    sprinters_gold = summer_olympics[(summer_olympics['Medal']=='Gold') & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    sprinters_gold
    sprinters_gold_Weight = sprinters_gold.groupby(['Year'])['Weight'].mean()

    #sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin('Gold','Silver','Bronze')
                                          #     & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    med=['Gold','Silver','Bronze']
    sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin(med)) & (summer_olympics['Event']=="Athletics Men's 100 metres")]

    sprinters_medal
    sprinters_medal_Weight = sprinters_medal.groupby(['Year'])['Weight'].mean()

    sprint = pd.merge(sprinters_Weight, sprinters_gold_Weight, on=['Year']).reset_index()

    sprint = pd.merge(sprint, sprinters_medal_Weight, on=['Year'])

    sprint.columns = ['Year', 'Overall Participants Average', 'Gold Medalists', 'All Medalists']
    #Athletics Men's 100 metres
    return sprint


# In[16]:


def sprinter_age(summer_olympics):
    """
    >>> sprinter_age()
    Traceback (most recent call last):
    ...
    TypeError: sprinter_age() missing 1 required positional argument: 'summer_olympics'

    >>> sprinter_age('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> sprinter_age(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> sprinter_age('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: sprinter_age() takes 1 positional argument but 3 were given

    >>> sprinter_age(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined



    :param summer_olympics:
    :return:
    """

    sprinters = summer_olympics[summer_olympics['Event']=="Athletics Men's 100 metres"]
    sprinters_Age  = sprinters.groupby(['Year'])['Age'].mean()

    sprinters_gold = summer_olympics[(summer_olympics['Medal']=='Gold') & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    sprinters_gold
    sprinters_gold_Age = sprinters_gold.groupby(['Year'])['Age'].mean()

    #sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin('Gold','Silver','Bronze')
                                          #     & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    med=['Gold','Silver','Bronze']
    sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin(med)) & (summer_olympics['Event']=="Athletics Men's 100 metres")]

    sprinters_medal
    sprinters_medal_Age = sprinters_medal.groupby(['Year'])['Age'].mean()

    sprint = pd.merge(sprinters_Age, sprinters_gold_Age, on=['Year']).reset_index()

    sprint = pd.merge(sprint, sprinters_medal_Age, on=['Year'])

    sprint.columns = ['Year', 'Overall Participants Average', 'Gold Medalists', 'All Medalists']
    #Athletics Men's 100 metres
    return sprint


# In[17]:


def sprinter_bmi(summer_olympics):
    """
    >>> sprinter_bmi()
    Traceback (most recent call last):
    ...
    TypeError: sprinter_bmi() missing 1 required positional argument: 'summer_olympics'

    >>> sprinter_bmi('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: string indices must be integers

    >>> sprinter_bmi(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> sprinter_bmi('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: sprinter_bmi() takes 1 positional argument but 3 were given

    >>> sprinter_bmi(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined

    :param summer_olympics:
    :return:
    """

    sprinters = summer_olympics[summer_olympics['Event']=="Athletics Men's 100 metres"]
    sprinters_BMI  = sprinters.groupby(['Year'])['BMI'].mean()

    sprinters_gold = summer_olympics[(summer_olympics['Medal']=='Gold') & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    sprinters_gold
    sprinters_gold_BMI = sprinters_gold.groupby(['Year'])['BMI'].mean()

    #sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin('Gold','Silver','Bronze')
                                          #     & (summer_olympics['Event']=="Athletics Men's 100 metres")]
    med=['Gold','Silver','Bronze']
    sprinters_medal = summer_olympics[(summer_olympics['Medal'].isin(med)) & (summer_olympics['Event']=="Athletics Men's 100 metres")]

    sprinters_medal
    sprinters_medal_BMI = sprinters_medal.groupby(['Year'])['BMI'].mean()

    sprint = pd.merge(sprinters_BMI, sprinters_gold_BMI, on=['Year']).reset_index()

    sprint = pd.merge(sprint, sprinters_medal_BMI, on=['Year'])

    sprint.columns = ['Year', 'Overall Participants AverBMI', 'Gold Medalists', 'All Medalists']
    #Athletics Men's 100 metres
    return sprint


# In[ ]:



# In[18]:


def medal_predictor(olympics,world_gdp,world_population,medalTotal_gdp):

    """
    >>> medal_predictor()
    Traceback (most recent call last):
    ...
    TypeError: medal_predictor() missing 4 required positional arguments: 'olympics', 'world_gdp', 'world_population', and 'medalTotal_gdp'

    >>> medal_predictor('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: medal_predictor() missing 3 required positional arguments: 'world_gdp', 'world_population', and 'medalTotal_gdp'

    >>> medal_predictor(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> medal_predictor('unknown.dataframe', 'country', 'gdp', 'population')
    Traceback (most recent call last):
    ...
    AttributeError: 'str' object has no attribute 'merge'

    >>> medal_predictor(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined





    :param olympics:
    :param world_gdp:
    :param world_population:
    :param medalTotal_gdp:
    :return:
    """

    olympics = olympics.merge(world_gdp,
                                                    left_on = ['NOC', 'Year'],
                                                    right_on = ['Country Code', 'Year'],
                                                    how = 'left')

    olympics = olympics.merge(world_population,
                                                 left_on = ['NOC', 'Year'],
                                                 right_on= ['Country Code', 'Year'],
                                                 how = 'left')
    olympics = olympics.loc[olympics['Season'] == "Summer", :]


    #print(olympics.size)


    olympic_team_gender = olympics.loc[:,['Year','Team', 'Name', 'Sex']].drop_duplicates()
    #print(olympic_team_gender.size)
    count_olympics_team_gender = pd.pivot_table(olympic_team_gender,
                                            index = ['Year', 'Team'],
                                            columns = 'Sex',
                                            aggfunc = 'count').reset_index()

    # rename columns as per column names in the 0th level
    count_olympics_team_gender.columns = count_olympics_team_gender.columns.get_level_values(0)

    # rename the columns appropriately
    count_olympics_team_gender.columns = ['Year', 'Team', 'Female_Athletes', 'Male_Athletes']
    count_olympics_team_gender = count_olympics_team_gender.fillna(0)

    # get total athletes per team-year
    count_olympics_team_gender['Total_Athletes'] = count_olympics_team_gender['Female_Athletes'] +     count_olympics_team_gender['Male_Athletes']

    year_team_contingent = count_olympics_team_gender.loc[:, ['Year', 'Team','Total_Athletes']]
    year_team_contingent.head()
    medalTotal_gdp.head()
    year_team_pop = olympics.loc[:, ['Year', 'Team', 'Population']].drop_duplicates()
    year_team_pop.head()
    medalTotal_gdp = medalTotal_gdp.drop('Year', 1)
    medalTotal_gdp.rename(columns={'Year_x': 'Year'}, inplace=True)





    medalTotal_gdp.drop_duplicates(['Year','Team'],keep= 'last' , inplace=True)
    medalTotal_gdp.head()

    year_team_pop.drop_duplicates(['Year','Team'],keep= 'last'  , inplace=True)
    #year_team_pop.head()
    medal_gdp_population = medalTotal_gdp.merge(year_team_pop,
                                                left_on = ['Year', 'Team'],
                                                right_on = ['Year', 'Team'],
                                                how = 'left'
                                                )

    medal_gdp_pop_contingent = medal_gdp_population.merge(year_team_contingent,
                                                         left_on = ['Year', 'Team'],
                                                         right_on = ['Year', 'Team'],
                                                         how = 'left')
    medal_gdp_pop_contingent.head()

    lin_model_data = medal_gdp_pop_contingent.loc[(medal_gdp_pop_contingent['Year'] > 1960), :]

    lin_model_data['GDP_per_capita'] = lin_model_data['GDP']/lin_model_data['Population']
    lin_model_data.dropna(how = 'any', inplace = True)

    lin_model_data.head()
    lin_model_data['Log_Population'] = np.log(lin_model_data['Population'])
    lin_model_data['Log_GDP'] = np.log(lin_model_data['GDP'])
    y, X = dmatrices('Medal_Won_Corrected ~ Log_GDP + Log_Population + Total_Athletes + GDP_per_capita',data = lin_model_data,return_type = 'dataframe')

    model = sm.OLS(y, X)
    result = model.fit()

    summary = result.summary()

    #from sklearn import metrics

    y_predicted = result.predict(X)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_predicted))
    return summary,rmse


# In[ ]:


def participants_edition(olympics,edition):
    """
    >>> participants_edition()
    Traceback (most recent call last):
    ...
    TypeError: participants_edition() missing 2 required positional arguments: 'olympics' and 'edition'

    >>> participants_edition('unknown.dataframe')
    Traceback (most recent call last):
    ...
    TypeError: participants_edition() missing 1 required positional argument: 'edition'

    >>> participants_edition(PandasDF)
    Traceback (most recent call last):
    ...
    NameError: name 'PandasDF' is not defined

    >>> participants_edition('unknown.dataframe', 'country','host')
    Traceback (most recent call last):
    ...
    TypeError: participants_edition() takes 2 positional arguments but 3 were given

    >>> participants_edition(df, unknown_argument , another )
    Traceback (most recent call last):
    ...
    NameError: name 'df' is not defined


    :param olympics:
    :return:
    """
    edition_olympics = olympics[(olympics.Season == edition)]
    sum = edition_olympics.groupby(['Year'])['ID'].count().reset_index()




    fig = plt.figure(figsize=(13,16))
    plt.subplot(211)
    ax = sns.pointplot(x = sum["Year"] , y = sum["ID"],markers="h")
    plt.xticks(rotation = 60)

    plt.ylabel("Participants Count")
    plt.title(edition+ " Olympics")


# In[ ]:



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



