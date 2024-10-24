#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


matches = pd.read_csv(r"Desktop\IPL_Matches_2022.csv")


# In[3]:


balls = pd.read_csv(r"Desktop\IPL_Ball_by_Ball_2022.csv")


# In[4]:


matches.head()


# In[5]:


balls.head()


# In[6]:


matches.shape


# In[7]:


balls.shape


# In[8]:


matches.info()
matches.describe()


# In[9]:


balls.info()
balls.describe()


# In[10]:


matches.drop(["method"],axis=1,inplace=True)


# In[11]:


matches.head()


# In[12]:


balls["extra_type"].fillna("Normal Delivery",inplace=True)
balls["player_out"].fillna('No One Out',inplace=True)
balls["kind"].fillna(0,inplace=True)
balls["fielders_involved"].fillna(0,inplace=True)


# In[13]:


balls.head()


# In[14]:


balls['batsman_run'] = pd.to_numeric(balls['batsman_run'], errors='coerce')
batter_runs = balls.groupby('batter')['batsman_run'].sum().reset_index()
batter_runs_sorted = batter_runs.sort_values(by='batsman_run', ascending=False)
most_runs_batter = batter_runs_sorted.iloc[0]
print("Batter with the most runs:")
print(most_runs_batter)


# In[18]:


import pandas as pd
balls['batsman_run'] = pd.to_numeric(balls['batsman_run'], errors='coerce')
batter_runs = balls.groupby('batter')['batsman_run'].sum().reset_index()
batter_runs_sorted = batter_runs.sort_values(by='batsman_run', ascending=False)
bottom_batters = batter_runs_sorted.tail(40)
print("Bottom 12 batters based on total runs scored:")
print(bottom_batters.head)


# In[19]:


bowler=balls[balls['kind']!=0]['bowler'].value_counts().reset_index(name='No. of wickets')
bowler.rename(columns={'index':'Bowler'}, inplace=True)
bowler.head(1)


# In[20]:


import pandas as pd
bowler_wickets = balls.groupby('bowler')['isWicketDelivery'].sum().reset_index()
bowler_wickets_sorted = bowler_wickets.sort_values(by='isWicketDelivery', ascending=True)
top_bowlers = bowler_wickets_sorted.head(21)
print("Top 21 bowlers with the least number of wicket deliveries:")
print(top_bowlers)


# In[21]:


bowler5=bowler.head(5)
bowler5


# In[22]:


list(balls['kind'].unique())


# In[23]:


wicket_keeper_catch=balls[balls['kind']=='caught']['fielders_involved'].value_counts().reset_index()
wicket_keeper_catch.rename(columns={'index':'wicket_keeper','fielders_involved':'No. of catches takes'}, inplace=True)
wicket_keeper_catch.head(5)


# In[24]:


wicket_keeper_runout=balls[balls['kind']=='run out']['fielders_involved'].value_counts().reset_index()
wicket_keeper_runout.rename(columns={'index':'wicket_keeper','fielders_involved':'run out'}, inplace=True)
wicket_keeper_runout.head(5)


# In[25]:


toss_won_count=pd.DataFrame({"Toss Won":matches['TossWinner']}).value_counts()
labels=[x[0]for x in toss_won_count.keys()]
toss_won_count


# In[26]:


matches.WonBy.unique()


# In[27]:


balls.groupby('batter').apply(lambda x: x[x["overs"].isin([1,2,3,4,5,6])]['total_run'].sum()).nlargest(1)
     


# In[28]:


balls.groupby('batter').apply(lambda x: x[x["overs"].isin([16,17,18,19,20])]['total_run'].sum()).nlargest(1)
     


# In[29]:


balls.groupby('batter').apply(lambda x: x[x["overs"].isin([19,20])]['total_run'].sum()).nlargest(1)


# In[30]:


balls.groupby('BattingTeam').apply(lambda x: x[x["overs"].isin([19,20])]['total_run'].sum()).nlargest(1)
     


# In[31]:


list(matches['WinningTeam'][matches.apply(lambda x:x['TossWinner']==x['WinningTeam'],axis=1)].unique())
     


# In[32]:


tosseswinmatches = matches['TossWinner'] == matches['WinningTeam']
count_true_false = tosseswinmatches.value_counts()

# Display the count of True and False occurrences
print(count_true_false)


# In[33]:


winning_teams=matches['WinningTeam'].value_counts().rename_axis("Team").reset_index(name='winning_count').head(10)
winning_teams


# In[34]:


sixes=balls.groupby('batter').apply(lambda x: x[x["batsman_run"]==6]['batsman_run'].sum()).nlargest(1)
sixes


# In[35]:


fours=balls.groupby('batter').apply(lambda x: x[x["batsman_run"]==4]['batsman_run'].sum()).nlargest(1)
fours


# In[36]:


balls.groupby('BattingTeam').apply(lambda x: x[x["batsman_run"]==6]['batsman_run'].sum()).nlargest(1)


# In[37]:


balls.groupby('BattingTeam').apply(lambda x: x[x["batsman_run"]==4]['batsman_run'].sum()).nlargest(1)


# In[38]:


duo=balls.groupby(['ID','batter','non-striker'])['total_run'].sum().reset_index()
duo_50=duo[duo['total_run']>50 ].sort_values(by=['total_run'],ascending=False)
duo_50


# In[39]:


duo_100=duo[duo['total_run']>=100]
duo_100


# In[40]:


toss_decision = matches['TossDecision'].value_counts()
toss_decision.head()


# In[41]:


toss_decision = matches.groupby('TossWinner')['TossDecision'].value_counts().sort_index()
toss_decision


# In[42]:


city_df = matches['City'].value_counts().reset_index()
city_df.columns = ['City', 'matches']

print(city_df)


# In[43]:


runs=balls.groupby(['batter'])['batsman_run'].sum().reset_index()
runs.columns=['Batsman','runs']
y=runs.sort_values(by='runs',ascending=False).head(3).reset_index().drop('index',axis=1)
y


# In[44]:


ball=balls.groupby(['bowler'])['isWicketDelivery'].sum().reset_index()
ball.columns=['Bowler','wicket']
y=ball.sort_values(by='wicket',ascending=False).head(3).reset_index().drop('index',axis=1)
y


# In[47]:


batsman_strike_rate = balls.groupby('batter').apply(lambda x: (x['batsman_run'].sum() / len(x)) * 100).reset_index()
batsman_strike_rate.columns = ['Batsman', 'StrikeRate']


# In[48]:


batsman_strike_rate_sorted = batsman_strike_rate.sort_values(by='StrikeRate', ascending=False)

print(batsman_strike_rate_sorted)


# In[49]:


bowler_economy_rate = balls.groupby('bowler').apply(lambda x: (x['total_run'].sum() / len(x)) * 6).reset_index()
bowler_economy_rate.columns = ['Bowler', 'EconomyRate']


# In[50]:


bowler_economy_sorted = bowler_economy_rate.sort_values(by='EconomyRate', ascending=False)
print(bowler_economy_sorted.tail())


# In[51]:


partnerships = balls.groupby(['batter', 'non-striker'])['total_run'].sum().reset_index()
partnerships


# In[53]:


balls['kind'] = balls['kind'].replace(0, 'not out')


# In[54]:


dismissals = balls.groupby('kind')['player_out'].count().reset_index()
dismissals


# In[55]:


team_performance = matches.groupby(['Season', 'Team2'])['WinningTeam'].count().reset_index()
team_performance


# In[56]:


team_performance1 = matches.groupby(['Season', 'Team1'])['WinningTeam'].count().reset_index()
team_performance1


# In[57]:


player_performance = balls.groupby(['ID', 'batter'])['batsman_run'].sum().reset_index()
player_performance


# In[58]:


all_rounders = balls.groupby('batter')['batsman_run'].sum().reset_index()
all_rounders['WicketsTaken'] = balls.groupby('bowler')['isWicketDelivery'].sum().reset_index()['isWicketDelivery']
all_rounders


# In[59]:


if 'batter' in balls.columns and 'bowler' in balls.columns:
    # Identify all-rounders
    batsman_runs = balls.groupby('batter')['batsman_run'].sum().reset_index()
    bowler_wickets = balls.groupby('bowler')['isWicketDelivery'].sum().reset_index()
    bowler_wickets.columns = ['batter', 'WicketsTaken']
    
    all_rounders = batsman_runs.merge(bowler_wickets, on='batter')
    
    weight_runs = 0.6
    weight_wickets = 0.4
    
    composite_scores = []
    for index, row in all_rounders.iterrows():
        runs = row['batsman_run']
        wickets = row['WicketsTaken']
        if runs > 0 or wickets > 0:
            composite_score = (runs * weight_runs) + (wickets * weight_wickets)
        else:
            composite_score = 0
        composite_scores.append(composite_score)
    
    all_rounders['CompositeScore'] = composite_scores
    all_rounders_sorted = all_rounders.sort_values(by='CompositeScore', ascending=False)
    
    print("Top All-rounders:")
    print(all_rounders_sorted.head(30))
else:
    print("Missing required columns for all-rounder analysis.")


# In[64]:


class TeamPerformanceAnalysis:
    def __init__(self, matches, balls):
        self.matches = matches
        self.balls = balls
        
    def get_team_stats(self, team):
        # Filter matches where the team is either batting or bowling
        team_matches = self.matches[(self.matches['Team1'] == team) | (self.matches['Team2'] == team)]
        
        # Calculate total runs scored by the team
        total_runs = self.balls[self.balls['BattingTeam'] == team]['total_run'].sum()
        
        
        
        # Calculate win-loss ratio
        total_matches = len(team_matches)
        total_wins = len(team_matches[team_matches['WinningTeam'] == team])
        win_loss_ratio = total_wins / total_matches if total_matches > 0 else 0
        
        return {
            'Team': team,
            'TotalRuns': total_runs,
          
            'TotalMatches': total_matches,
            'TotalWins': total_wins,
            'WinLossRatio': win_loss_ratio
        }



team_analysis = TeamPerformanceAnalysis(matches, balls)
team_stats = team_analysis.get_team_stats('Mumbai Indians')
print("Team Stats:", team_stats)


# In[65]:


class TeamPerformanceAnalysis:
    def __init__(self, matches, balls):
        self.matches = matches
        self.balls = balls
        
    def get_team_stats(self, team):
        # Filter matches where the team is either batting or bowling
        team_matches = self.matches[(self.matches['Team1'] == team) | (self.matches['Team2'] == team)]
        
        # Calculate total runs scored by the team
        total_runs = self.balls[self.balls['BattingTeam'] == team]['total_run'].sum()
        
        
        
        # Calculate win-loss ratio
        total_matches = len(team_matches)
        total_wins = len(team_matches[team_matches['WinningTeam'] == team])
        win_loss_ratio = total_wins / total_matches if total_matches > 0 else 0
        
        return {
            'Team': team,
            'TotalRuns': total_runs,
          
            'TotalMatches': total_matches,
            'TotalWins': total_wins,
            'WinLossRatio': win_loss_ratio
        }



team_analysis = TeamPerformanceAnalysis(matches, balls)
team_stats = team_analysis.get_team_stats('Chennai Super Kings')
print("Team Stats:", team_stats)


# In[66]:


class TeamPerformanceAnalysis:
    def __init__(self, matches, balls):
        self.matches = matches
        self.balls = balls
        
    def get_team_stats(self, team):
        # Filter matches where the team is either batting or bowling
        team_matches = self.matches[(self.matches['Team1'] == team) | (self.matches['Team2'] == team)]
        
        # Calculate total runs scored by the team
        total_runs = self.balls[self.balls['BattingTeam'] == team]['total_run'].sum()
        
        
        
        # Calculate win-loss ratio
        total_matches = len(team_matches)
        total_wins = len(team_matches[team_matches['WinningTeam'] == team])
        win_loss_ratio = total_wins / total_matches if total_matches > 0 else 0
        
        return {
            'Team': team,
            'TotalRuns': total_runs,
          
            'TotalMatches': total_matches,
            'TotalWins': total_wins,
            'WinLossRatio': win_loss_ratio
        }



team_analysis = TeamPerformanceAnalysis(matches, balls)
team_stats = team_analysis.get_team_stats('Rajasthan Royals')
print("Team Stats:", team_stats)


# In[68]:


class TeamPerformanceAnalysis:
    def __init__(self, matches, balls):
        self.matches = matches
        self.balls = balls
        
    def get_team_stats(self, team):
        # Filter matches where the team is either batting or bowling
        team_matches = self.matches[(self.matches['Team1'] == team) | (self.matches['Team2'] == team)]
        
        # Calculate total runs scored by the team
        total_runs = self.balls[self.balls['BattingTeam'] == team]['total_run'].sum()
        
        
        
        # Calculate win-loss ratio
        total_matches = len(team_matches)
        total_wins = len(team_matches[team_matches['WinningTeam'] == team])
        win_loss_ratio = total_wins / total_matches if total_matches > 0 else 0
        
        return {
            'Team': team,
            'TotalRuns': total_runs,
          
            'TotalMatches': total_matches,
            'TotalWins': total_wins,
            'WinLossRatio': win_loss_ratio
        }



team_analysis = TeamPerformanceAnalysis(matches, balls)
team_stats = team_analysis.get_team_stats('Gujarat Titans')
print("Team Stats:", team_stats)


# In[70]:


team_runs = balls.groupby('BattingTeam')['total_run'].sum()
if len(team_runs) == 0:
    print("No data available for analysis.")
elif len(team_runs) == 1:
    print("Only one team played in the match.")
else:

    max_runs_team = team_runs.idxmax()
    max_runs = team_runs.max()

   
    min_runs_team = team_runs.idxmin()
    min_runs = team_runs.min()

    print("Team '{}' scored the most runs: {}".format(max_runs_team, max_runs))
    print("Team '{}' scored the least runs: {}".format(min_runs_team, min_runs))


# In[ ]:




