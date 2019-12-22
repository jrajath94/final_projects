**Topic: Analyzing Olympic Data Set from 1896 to 2016**

Main Notebook - https://nbviewer.jupyter.org/github/jrajath94/final_projects/blob/master/Final%20Submission%20-%20Olympics.ipynb

Functions - https://github.com/jrajath94/final_projects/blob/master/Olympic_Functions.py

Hypothesis 1:  Does Home team hold advantage compared to foreign teams?

Hypothesis 2: More the number of events, more the participation of athletes

Hypothesis 3: Does GDP play a role in determining the medals count won by a country?

Hypothesis 4: Does the geographic location of a country gives them advantage in Olympics?

Overview: We are analyzing Olympic data set from year 1896 to 2016. Our first data set includes details of the athlete, details of the sport event, the country where the Olympic took place, medal won in a particular event, the season in which the Olympic took place i.e. Summer or Winter. Our second data set includes details related to GDP of a country. Our third data set includes details related to population of a country.

Data Cleaning:

In the dataset the athletes who had not won any events had their value as N/A. To make it easier to understand, we are replacing it with the value as &#39;Did Not Win&#39;.

Next, we are diving the events into team event and individual event. The reason for doing this is that, If a Country has won medal in soccer, it should count one single medal instead of 11 medals (since 11 players in a team).

Lastly, there were events where the athletes who participated were dead but their age was still updating. So, we removed events whose name started from &quot;Arts&quot;



Analysis of Hypothesis:

We are analyzing the following details:

1 - Does home team in an Olympic event holds an advantage?

Conclusion: From the above screenshot, we can come to conclusion that home team definitely has an advantage in Olympic event.

2 - More the number of events, more the participation of athletes.



Conclusion: As the population keeps on increasing, the interest of people increasing when it comes to their own liking for the sport. As a result, the events in Olympics over the years are increasing, which is motivation more and more people to participate in the Olympics.



3 - Does GDP play a role in determining the medals count won by a country?

Conclusion: GDP is a proxy for a country&#39;s resources. A higher GDP means more resources to allocate to sports!

The plot shows a 0.5765 correlation between GDP and medals won! That&#39;s a significant correlation. So GDP positively impacts the number of medals won by a team.





4 - Does the geographic location of a country gives them advantage in Olympics?



Conclusion: From the above result, we can see that apart from countries like USA, Germany and Russia, when an Olympic is hosted in winter season, countries like Finland, Norway has more number of medals won in comparison with Summer Olympics. We can come to conclusion, that the geographic conditions, do hold some advantage in determining the chances of a particular country winning medals.



Apart from the hypothesis stated above, different analysis where also taken into consideration, by taking the past data into account.

1 - Most Popular sports performing event in a country?

Every country has atlas one event in which it is performing better overtime and winning medals.

In our scenario, we have taken top 3 performing countries (US, Germany, Russia) and we are identifying the top performing sports event by the number of gold medals won in the events.



From the result above, you can see that Germany&#39;s top performing event as Luge Men&#39;s Event, Russia&#39;s top performing Event as Gymnastics Women&#39;s Uneven Bars, USA top performing event is Athletics Men&#39;s Long Jump.

2 - Changing Trends in Women Participation in Olympics



Conclusion: Women athlete participation increased over the time from the time it started. We can see a significant change in participation from 26.1% at Seoul in 1988 to 45.2% at Rio in 2016. Eventually, a near balanced male female participation was observed. Growing participation is because of various factors like support from IOC, growing awareness about the Olympics and compulsory women events added. 

3 - Does a player who wins a medal holds a physiological advantage?

As you can see from below image, we have calculated the average height by taking all athletes into account in a particular year.

Next we have calculated the average height of athletes who have won gold medals.

At Last, we have calculated the average height of athletes who have won gold, silver and bronze medals.



Conclusion: From the result above, we see that height of Gold Medalists is more in comparison to mean height of all athletes and mean height of all athletes who have won atlas one medal.

4 - Analyzing the distribution of Age vs height in olympics over the years?

 Here , we see as to how the Age and height of athletes have evolved over the years .



5 - Analyzing the BMI distribution in olympics over the years?

Here , we analyse BMI of athletes from different sports .
