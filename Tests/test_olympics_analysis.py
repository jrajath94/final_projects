from unittest import TestCase
from Latest_Submission import *
from Tests.submission_test import *

world_gdp_test = read_data_df('Data/world_gdp.csv', skiprows=3)
noc_country_test = read_data_df('Data/noc_regions.csv')
world_population_test = read_data_df('Data/world_pop.csv')
olympics_test = read_data_df('Data/athlete_events.csv')
olympics_host_test = read_data_df('Data/olm2.csv')
world_hdi_test = read_data_df('Data/hdi.csv')

olympics_test1 = Olympics_analysis.oly_medal(olympics_test)
olympics_test1 = Olympics_analysis.one_hot_encoding(olympics_test1)
world = Olympics_analysis.world_gdp(world_gdp_test)
olympics_NOC = Olympics_analysis.noc_merge(olympics_test, noc_country_test)
olympics_host, olympic_years = Olympics_analysis.olympics_host_years(
    olympics_host_test)
world_population_columns = list(world_population_test.columns.values)[4:]
world_population = Olympics_analysis.filter_years(
    world_population_test, world_population_columns, olympic_years)
world_gdp = Olympics_analysis.filter_years(
    world_gdp_test, world_population_columns, olympic_years)
world_gdp_columns = list(world_gdp.columns.values)[4:]

team_events = Olympics_analysis.team(olympics_NOC)
medal_tally_1 = Olympics_analysis.tally_medal(olympics_NOC)
medal_tally_by_year = Olympics_analysis.tally_medal_year(medal_tally_1)
medal_tally_overall = Olympics_analysis.tally_medal_overall(medal_tally_1)
olympic_host_noc = Olympics_analysis.olympic_host_noc(
    olympics_host_test, noc_country_test)


class TestOlympics_analysis(TestCase):

    def test_oly_medal(self):
        olympics_test1 = Olympics_analysis.oly_medal(olympics_test)
        s = olympics_test1.loc[(olympics_test1.Name == 'Minna Maarit Aalto') & (
            olympics_test1.Year == 2000)]['Medal'].values
        assert(str(s) == "['No_Medal']")

    def test_one_hot_encoding(self):
        s = olympics_test1.loc[(olympics_test1.Name == 'Edgar Lindenau Aabye') & (
            olympics_test1.Year == 1900)]['Gold_Medal'].values
        t = olympics_test1.loc[(olympics_test1.Name == 'A Dijiang') & (
            olympics_test1.Year == 1992)]['No_Medal'].values
        u = olympics_test1.loc[(olympics_test1.Name == 'A Lamusi') & (
            olympics_test1.Year == 2012)]['Bronze_Medal'].values
        v = olympics_test1.loc[(olympics_test1.Name == 'Tomasz Ireneusz ya') & (
            olympics_test1.Year == 1998)]['Silver_Medal'].values
        assert(str(s) == "[1]")
        assert(str(t) == "[1]")
        assert(str(u) == "[0]")
        assert(str(v) == "[0]")

    def test_world_gdp(self):
        s = Olympics_analysis.world_gdp(world_gdp_test)
        s = world.shape[1]
        assert(str(s) == '59')

    def test_noc_merge(self):
        s = Olympics_analysis.noc_merge(olympics_test, noc_country_test)

        assert(str(s.shape[0]) == '271116')

    def test_olympics_host_years(self):
        s = olympics_host.loc[(olympics_host['Host City'] == 'Athens') & (
            olympics_host.Year == 1896)]['Winter'].values
        assert(str(s) == "['No_Olympics']")

    def test_team(self):
        team_events = Olympics_analysis.team(olympics_NOC)
        assert(str(team_events.shape[0]) == "244")

    def test_tally_medal(self):
        medal_tally_1 = Olympics_analysis.tally_medal(olympics_NOC)
        assert (str(medal_tally_1.shape[0]) == "15467")

    def test_tally_medal_year(self):
        s = Olympics_analysis.tally_medal_year(medal_tally_1)
        s = medal_tally_by_year.reset_index()
        s = s.reset_index().loc[(s.NOC == 'YUG') & (
            s.Year == 1988)]['Tally_Overall'].values
        assert(str(s) == "[11]")

    def test_tally_medal_overall(self):
        s = Olympics_analysis.tally_medal_overall(medal_tally_1)
        s = medal_tally_overall.reset_index()
        s = s.reset_index().loc[(s.NOC == 'GBR')]['Tally_Overall'].values
        assert(str(s) == "[863]")

    def test_previous_year(self):
        df1 = pd.DataFrame({'a': [1, 5, 3, 9, 5]})
        s = Olympics_analysis.previous_year(df1['a'])
        assert(str(s) == '[0. 1. 5. 3. 9.]')

    def test_next_year(self):
        df1 = pd.DataFrame({'a': [1, 5, 3, 9, 5]})
        s = Olympics_analysis.previous_year(df1['a'])
        assert(str(s) == '[0. 1. 5. 3. 9.]')

    def test_getroutedistance(self):
        df1 = pd.DataFrame({'a': [22.5, 32.5, 45.3, -12.9]})
        df2 = pd.DataFrame({'b': [98.6, -34.8, 76.2, 67.7]})
        s = Olympics_analysis.getroutedistance(df1['a'], df2['b'])
        assert (
            str(s) == '[0, 12167.881524061015, 8943.850746396238, 6501.6404508618925]')
