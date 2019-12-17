#!/usr/bin/env python
# coding: utf-8

# In[25]:


#import pandas as pd
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
#import modin.pandas as pd


#df_list = [pd.DataFrame() for df in df_names]

#Merge_olympics,merge_olympics_countrycode,olympics_merge_gdp,olympics_new = pd.DataFrame()

# def read_data_modin(file, **kwargs):
    
#     filename = mpd.read_csv(file, **kwargs)
#     return filename 



 



def read_data(file, **kwargs):
    
    filename = pd.read_csv(file, **kwargs)
#     df = ddf.read_csv(file,**kwargs, blocksize=1000000,)
#     df = df.compute(scheduler='processes')
#     return df 
    return filename

def read_data_df(file, **kwargs):
    
   # filename = pd.read_csv(file, **kwargs)
    df = ddf.read_csv(file,**kwargs, blocksize=1000000)
    df = df.compute(scheduler='processes')
    return df 
    #return filename

def get_shape(dataframe):
    return dataframe.shape

def get_stats(dataframe):
    return dataframe.info()

def get_missing_values(dataframe):
    return dataframe.isnull().sum()

def get_info(dataframe):
   # list(filter(None, test_list)) 
    columns =  list(filter(None,list(dataframe.columns.values)))
    #columns = [i for i in columns if i]
    names = str(columns)[1:-1]
    row, col = dataframe.shape
    print("\nThe Dataframe has {} rows and {} columns".format(row,col))
    
    print("Column Names - {} \n".format(names))

    


# In[2]:


def data_cleaning(olympics, noc_country,world_gdp , world_population, olympics_host):

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


# In[3]:


from numba import jit

@jit(nopython=True)
def numbaeq(flatdata,x,nrow,ncol,data):
  dtype=data.dtype
  size=ncol*nrow
  ix=np.empty(size,dtype=dtype)
  jx=np.empty(size,dtype=dtype)
  count=0
  k=0
  while k<size:
    if flatdata[k]==x :
      ix[count]=k//ncol
      jx[count]=k%ncol
      count+=1
    k+=1          
  return ix[:count],jx[:count]

def whereequal(data,x): 
    return numbaeq(data.ravel(),x,*data.shape,data)


# In[4]:


def correlation( Merge_olympics,merge_olympics_countrycode,olympics_merge_gdp,olympics_new):


    get_ipython().run_line_magic('pylab', 'inline')
    olympics_merge = olympics_new.loc[(olympics_new['Year'] > 1960) & (olympics_new['Season'] == "Summer"), :]

    # Reset row indices
    olympics_merge = olympics_merge.reset_index()
    olympics_host = pd.read_csv('olym.csv',encoding="ISO-8859–1")
    mergedframe = pd.merge(olympics, olympics_host, left_on='City', right_on='Host City', how ='inner')

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

    plot(medalTotal_gdp.loc[row5, 'GDP'], 
         medalTotal_gdp.loc[row5, 'Medal_Won_Corrected'] , 
         linestyle = 'none', 
         marker = 'x',
        alpha = 0.4 , color='red')
    xlabel('GDP of Country')

    ylabel('Count of Medals')
    text(np.nanpercentile(medalTotal_gdp['GDP'], 99.6), 
         max(medalTotal_gdp['Medal_Won_Corrected']) - 50,
         "Correlation = " + str(correlation))
    
    return mergedframe,medalTotal_gdp


# In[5]:


def team_performance(mergedframe, countries):
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
    


# In[6]:


def home_advantage(medal_tally_agnostic):

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

    # check rows where host country is the same as team
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


# In[7]:


def top_10_countries_summer(noc_country,olympics):
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


# In[24]:


def top_10_countries_winter(player):
    top10_winter = player[(player['Season']=='Winter') & (player['Medal']!='No Medal')].groupby('Country').count().reset_index()[['Country','Medal']].sort_values('Medal', ascending=False).head(10)
    f, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Country", y="Medal", data=top10_winter, label="Country", color="purple")

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2,p.get_height(),

                '{:1.0f}'.format(p.get_height()),
                ha="center")

    #ax.set_xlabel('Country', size=14, color="green")
    #ax.set_ylabel('Total Medals Won', size=14, color="green")
    ax.set_title('Top 10 countries with total medals in Winter Olympic games', size=16)
    plt.show()


# In[9]:


def athletes_summer(olympics):
    summerData = olympics[olympics['Season']=='Summer']

    summerAthletes = summerData.pivot_table(summerData, index=['Year'], aggfunc=lambda x: len(x.unique())).reset_index()[['Year','ID']]
    summerSports = summerData.groupby('Year')['Sport'].nunique().reset_index()
    summerEvents = summerData.groupby('Year')['Event'].nunique().reset_index()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(22,18))

    sns.barplot(x='Year', y='ID', data=summerAthletes, ax=ax[0], color="red")
    sns.barplot(x='Year', y='Sport', data=summerSports, ax=ax[1], color="blue")
    sns.barplot(x='Year', y='Event', data=summerEvents, ax=ax[2], color="orange")

    j = 0
    for i in ['Athletes', 'Sports', 'Events']:
        ax[j].set_xlabel('Year', size=14)
        ax[j].set_ylabel(i, size=14 )
        ax[j].set_title(i + ' in Summer Olympic ', size=18)
        j = j + 1

    for i in range(3):
        for p in ax[i].patches:
            ax[i].text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),
                    fontsize=12, color='black', ha='center', va='bottom')
    plt.show()


# In[10]:


def athletes_winter(olympics):
    winterData = olympics[olympics['Season']=='Winter']

    winterAthletes = winterData.pivot_table(winterData, index=['Year'], aggfunc=lambda x: len(x.unique())).reset_index()[['Year','ID']]
    winterSports = winterData.groupby('Year')['Sport'].nunique().reset_index()
    winterEvents = winterData.groupby('Year')['Event'].nunique().reset_index()

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(22,18))

    sns.barplot(x='Year', y='ID', data=winterAthletes, ax=ax[0], color="red")
    sns.barplot(x='Year', y='Sport', data=winterSports, ax=ax[1], color="blue")
    sns.barplot(x='Year', y='Event', data=winterEvents, ax=ax[2], color="orange")

    j = 0
    for i in ['Athletes', 'Sports', 'Events']:
        ax[j].set_xlabel('Year', size=14)
        ax[j].set_ylabel(i, size=14)
        ax[j].set_title(i + ' in Winter Olympic ', size=18)
        j = j + 1

    for i in range(3):
        for p in ax[i].patches:
            ax[i].text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()),
                    fontsize=12, color='black', ha='center', va='bottom')
    plt.show()


# In[11]:


def BMI_by_event_participants(olympics ,gender):
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





# In[12]:


def BMI_by_event_gold_medalists(olympics,gender):

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





# In[13]:


# Evolution Based on BMI
def BMI_by_time_gold_medalists(olympics):
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



# In[14]:


def participants(olympics):
    summer_olympics = olympics[(olympics.Season == 'Summer')]
#    summer_olympics.groupby(['Year'])['ID'].count().reset_index(drop=True)
    return summer_olympics


# In[15]:


def sprinter_Height(summer_olympics):
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


# In[16]:


def sprinter_weight(summer_olympics):
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


# In[17]:


def sprinter_age(summer_olympics):
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


# In[ ]:


def sprinter_bmi(summer_olympics):
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
    city_to_country = {'Tokyo': 'Japan',
                      'Mexico City': 'Mexico',
                      'Munich': 'Germany',
                      'Montreal': 'Canada',
                      'Moscow': 'Russia',
                      'Los Angeles': 'USA',
                      'Seoul': 'South Korea',
                      'Barcelona': 'Spain',
                      'Atlanta': 'USA',
                      'Sydney': 'Australia',
                      'Athens': 'Greece',
                      'Beijing': 'China',
                      'London': 'UK',
                      'Rio de Janeiro': 'Brazil'}

    # Map cities to countries
    olympics['Country_Host'] = olympics['City'].map(city_to_country)

    #print the 
    olympics.loc[:, ['Year', 'Country_Host']].drop_duplicates().sort_values('Year')




    year_host = olympics.loc[:, ['Year', 'Country_Host']].drop_duplicates()

    # merge this with the larger dataset
    lin_model_data = medal_gdp_pop_contingent.merge(year_host,
                                  left_on = 'Year',
                                  right_on = 'Year',
                                  how = 'left')

    lin_model_data = lin_model_data.loc[(lin_model_data['Year'] > 1960), :]

    lin_model_data['GDP_per_capita'] = lin_model_data['GDP']/lin_model_data['Population']
    lin_model_data.dropna(how = 'any', inplace = True)

    lin_model_data.head()
    lin_model_data['Log_Population'] = np.log(lin_model_data['Population'])
    lin_model_data['Log_GDP'] = np.log(lin_model_data['GDP'])
    y, X = dmatrices('Medal_Won_Corrected ~ Log_GDP + Log_Population + Total_Athletes + GDP_per_capita', 
                    data = lin_model_data,
                    return_type = 'dataframe')

    model = sm.OLS(y, X)
    result = model.fit()

    summary = result.summary()

    #from sklearn import metrics

    y_predicted = result.predict(X)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_predicted))
    return summary,rmse


# In[19]:


def participants_summer(olympics):

    summer_olympics = olympics[(olympics.Season == 'Summer')]
    sum = summer_olympics.groupby(['Year'])['ID'].count().reset_index()




    fig = plt.figure(figsize=(13,16))
    plt.subplot(211)
    ax = sns.pointplot(x = sum["Year"] , y = sum["ID"],markers="h")
    plt.xticks(rotation = 60)

    plt.ylabel("Participants Count")
    plt.title("Summer Olympics")


# In[20]:


def participants_winter(olympics):
    winter_olympics = olympics[(olympics.Season == 'Winter')]
    win = winter_olympics.groupby(['Year'])['ID'].count().reset_index()




    fig = plt.figure(figsize=(13,16))
    plt.subplot(211)
    ax = sns.pointplot(x = win["Year"] , y = win["ID"],markers="h")
    plt.xticks(rotation = 60)

    plt.ylabel("Participants Count")
    plt.title("Summer Olympics")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




