***Project : Analyzing Olympic Data Set from 1896 to 2016***

Main Notebook - [https://nbviewer.jupyter.org/github/jrajath94/final\_projects/blob/master/Latest\_Submission.ipynb](https://nbviewer.jupyter.org/github/jrajath94/final_projects/blob/master/Latest_Submission.ipynb)

Functions (with Doctests) - [https://github.com/jrajath94/final\_projects/blob/master/Latest\_Submission.py](https://github.com/jrajath94/final_projects/blob/master/Latest_Submission.py)

Unit Tests  - [https://github.com/jrajath94/final\_projects/blob/master/Tests/test\_olympics\_analysis.py](https://github.com/jrajath94/final_projects/blob/master/Tests/test_olympics_analysis.py)

Overview: We are analyzing Olympic data set from year 1896 to 2016. Our first data set includes details of the athlete, details of the sport event, the country where the Olympic took place, medal won in a particular event, the season in which the Olympic took place i.e. Summer or Winter. Our second data set includes details related to GDP of a country. Our third data set includes details related to population of a country.

**Data Cleaning:**

In the dataset the athletes who had not won any events had their value as N/A. To make it easier to understand, we are replacing it with the value as &#39;No Medal&#39;.

Next, we are diving the events into team event and individual event. The reason for doing this is that, If a Country has won medal in soccer, it should count one single medal instead of 11 medals (since 11 players in a team).

Analysis of Hypothesis:

We are analyzing the following details:

**1 - Does home team in an Olympic event holds an advantage?**

Here, we try to investigate whether Home advantage affects the medal tally of the country.

**2 - More the number of events, the more participation of athletes.**

Inference from the Project : As the population keeps on increasing, the interest of people increasing when it comes to their own liking for the sport. As a result, the events in the Olympics over the years are increasing, which is motivation more and more people to participate in the Olympics.

**3 -**  ** Does the geographic location of a country gives them advantage in Olympics?**

Inference from the Project: We can see that apart from countries like USA, Germany and Russia, when an Olympic is hosted in winter season, countries like Canada, South Korea  has more number of medals won in comparison with Summer Olympics. We can come to conclusion, that the geographic conditions, do hold some advantage in determining the chances of a particular country winning medals.

**4 - Does an athlete enjoy physiological advantage?**

Inference from the Project:

We can infer that:

a) - The average height of Gold Medalists is higher (in max years) in comparison to overall participants and participants who have won silver or bronze medal

b) - The average weight of Gold Medalists is higher (in max years) in comparison to overall participants and participants who have won silver or bronze medal

**5 -  Analyzing the medal won by all countries and which is the top performing country in the olympics:**

Inference from the Project:

we can infer that the top performing country in the olympics is USA.



**6 - Analyzing the trend of medals won by athletes of USA in olympics.**

Inference from the Project:

We can infer that over the duration more number of athletes have won Gold medals in comparison to silver and Bronze.

**7 - Is it possible to predict the medal Tally of a country?**

Here , we try to predict if we could estimate the medal tally of the country based on population , gdp , gdp\_per\_capita and later with HDI of the country . We&#39;ll also see as to how correlated these parameters are with the overall medal tally.

We can infer that using these factors we could predict the medal tally of a given country within an error of 4 medals per tally



**8 - Does hosting an Olympic have and impact on the GDP of the host country?**



It can be observed that the 3 years where Japan hosted the Olympics, GDP per capita observed a higher increase in comparison to the GDP per capita in the participating years

**9 - Analyzing Female trend over the years**

Inference from the Project:

Women athlete participation increased over the time from the time it started.

We can see a significant change in participation from 26.1% at Seoul in 1988 to 45.2% at Rio in 2016. Eventually,a near balanced male female participation was observed.

Growing participation is because of various factors like support from IOC, growing awareness about the Olympics and compulsory women events added.

**10 - Calculating top 3 PERFORMING events for USA,RUSSIA AND UK IN SUMMER OLYMPICS**

Here we investigate, the top performing event  of a given set of countries in Summer Olympics  which is based on the highest number of Gold Medals Won

**11 - Calculating top 3 PERFORMING events for USA,GERMANY AND NORWAY IN WINTER OLYMPICS**

Here we investigate, the top performing event  of a given set of countries in Winter Olympics  which is based on the highest number of Gold Medals Won
