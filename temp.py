import pandas as pd
import os
os.chdir("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MDataFiles_Stage1")

mevents2015 = pd.read_csv("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MEvents2015.csv")
mevents2016 = pd.read_csv("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MEvents2016.csv")
mevents2017 = pd.read_csv("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MEvents2017.csv")
mevents2018 = pd.read_csv("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MEvents2018.csv")
mevents2019 = pd.read_csv("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MEvents2019.csv")
mplayers = pd.read_csv("C:\\Users\\Shubham Kumar\\Desktop\\March Madness\\MPlayers.csv")

mevents2015 = pd.merge(mevents2015, mplayers, left_on="EventPlayerID", right_on="PlayerID", 
                       how='left').drop(["PlayerID","TeamID"], axis=1)

mevents2016 = pd.merge(mevents2016, mplayers, left_on="EventPlayerID", right_on="PlayerID", 
                       how='left').drop(["PlayerID","TeamID"], axis=1)

mevents2017 = pd.merge(mevents2017, mplayers, left_on="EventPlayerID", right_on="PlayerID", 
                       how='left').drop(["PlayerID","TeamID"], axis=1)

mevents2018 = pd.merge(mevents2018, mplayers, left_on="EventPlayerID", right_on="PlayerID", 
                       how='left').drop(["PlayerID","TeamID"], axis=1)

mevents2019 = pd.merge(mevents2019, mplayers, left_on="EventPlayerID", right_on="PlayerID", 
                       how='left').drop(["PlayerID","TeamID"], axis=1)

MTeams = pd.read_csv("MTeams.csv")
MGameCities                     = pd.read_csv('MGameCities.csv')                    
MMasseyOrdinals                 = pd.read_csv('MMasseyOrdinals.csv')                
MNCAATourneyCompactResults      = pd.read_csv('MNCAATourneyCompactResults.csv')     
MNCAATourneyDetailedResults     = pd.read_csv('MNCAATourneyDetailedResults.csv')    
MNCAATourneySeedRoundSlots      = pd.read_csv('MNCAATourneySeedRoundSlots.csv')     
MNCAATourneySeeds               = pd.read_csv('MNCAATourneySeeds.csv')              
MNCAATourneySlots               = pd.read_csv('MNCAATourneySlots.csv')              
MRegularSeasonCompactResults    = pd.read_csv('MRegularSeasonCompactResults.csv')   
MRegularSeasonDetailedResults   = pd.read_csv('MRegularSeasonDetailedResults.csv')  
MSeasons                        = pd.read_csv('MSeasons.csv')                       
MSecondaryTourneyCompactResults = pd.read_csv('MSecondaryTourneyCompactResults.csv')
MSecondaryTourneyTeams          = pd.read_csv('MSecondaryTourneyTeams.csv')         
MTeamCoaches                    = pd.read_csv('MTeamCoaches.csv')                   
MTeamConferences                = pd.read_csv('MTeamConferences.csv')               
MTeamSpellings                  = pd.read_csv('MTeamSpellings.csv', encoding = "ISO-8859-1")                 
Cities                          = pd.read_csv('Cities.csv')                         
Conferences                     = pd.read_csv('Conferences.csv')                    
MConferenceTourneyGames         = pd.read_csv('MConferenceTourneyGames.csv')

MRegularSeasonDetailedResults['id'] = MRegularSeasonDetailedResults.Season.map(str) + '_' + MRegularSeasonDetailedResults[['WTeamID','LTeamID']].min(axis = 1).map(str) + '_' +MRegularSeasonDetailedResults[['WTeamID','LTeamID']].max(axis = 1).map(str)
MRegularSeasonDetailedResults['min_id'] = MRegularSeasonDetailedResults[['WTeamID','LTeamID']].min(axis = 1)
MRegularSeasonDetailedResults['max_id'] = MRegularSeasonDetailedResults[['WTeamID','LTeamID']].max(axis = 1)


MNCAATourneyCompactResults['id'] = MNCAATourneyCompactResults.Season.map(str) + '_' + MNCAATourneyCompactResults[['WTeamID','LTeamID']].min(axis = 1).map(str) + '_' +MNCAATourneyCompactResults[['WTeamID','LTeamID']].max(axis = 1).map(str)
MNCAATourneyCompactResults['min_id'] = MNCAATourneyCompactResults[['WTeamID','LTeamID']].min(axis = 1)
MNCAATourneyCompactResults['max_id'] = MNCAATourneyCompactResults[['WTeamID','LTeamID']].max(axis = 1)

import numpy as np

MNCAATourneyCompactResults['Event'] = np.where(MNCAATourneyCompactResults.min_id == MNCAATourneyCompactResults.WTeamID,1,0)



a = pd.DataFrame(MRegularSeasonDetailedResults[['WTeamID','WScore','Season']].groupby(['WTeamID','Season'])['WScore'].agg('sum'))
a['id'] = a.index
a['TeamID'] = a['id'].apply(lambda x: x[0])
a['Season'] = a['id'].apply(lambda x: x[1])
a['id_1'] = a.Season.map(str) + '_' + a.TeamID.map(str) 

b = pd.DataFrame()
b = pd.DataFrame(MRegularSeasonDetailedResults[['LTeamID','LScore','Season']].groupby(['LTeamID','Season'])['LScore'].agg('sum'))
b['id'] = b.index
b['TeamID'] = b['id'].apply(lambda x: x[0])
b['Season'] = b['id'].apply(lambda x: x[1])
b['id_1'] = b.Season.map(str) + '_' + b.TeamID.map(str) 

team_agg = pd.merge(a,b[['LScore','id_1']],left_on='id_1',right_on='id_1',how='left')

a = pd.DataFrame(MRegularSeasonDetailedResults[['WTeamID','Season']].groupby(['WTeamID','Season'])['WTeamID'].agg('count'))
a['id'] = a.index
a['TeamID'] = a['id'].apply(lambda x: x[0])
a['Season'] = a['id'].apply(lambda x: x[1])
a['id_1'] = a.Season.map(str) + '_' + a.TeamID.map(str) 

team_agg = pd.merge(team_agg,a[['WTeamID','id_1']],left_on='id_1',right_on='id_1',how='left')

a = pd.DataFrame(MRegularSeasonDetailedResults[['LTeamID','Season']].groupby(['LTeamID','Season'])['LTeamID'].agg('count'))
a['id'] = a.index
a['TeamID'] = a['id'].apply(lambda x: x[0])
a['Season'] = a['id'].apply(lambda x: x[1])
a['id_1'] = a.Season.map(str) + '_' + a.TeamID.map(str) 

team_agg = pd.merge(team_agg,a[['LTeamID','id_1']],left_on='id_1',right_on='id_1',how='left')

team_agg = team_agg.rename(columns = {'WTeamID':'Matches_Won',
                                      'LTeamID': 'Matches_Lost'})

team_agg['Avg_Win_Points'] = team_agg.WScore/team_agg.Matches_Won
team_agg['Avg_Lose_Points'] = team_agg.LScore/team_agg.Matches_Lost
team_agg['Avg_Points'] = (team_agg.WScore + team_agg.LScore)/(team_agg.Matches_Won + team_agg.Matches_Lost)
team_agg['Matches_Played'] = team_agg.Matches_Won + team_agg.Matches_Lost
team_agg['Win_perc'] = team_agg.Matches_Won/team_agg.Matches_Played
team_agg['Lose_perc'] = 1 - team_agg.Win_perc


match_agg = pd.DataFrame()
a = pd.DataFrame(MRegularSeasonDetailedResults.loc[(MRegularSeasonDetailedResults.WTeamID == MRegularSeasonDetailedResults.min_id ),['id','WScore','Season']].groupby(['id','Season'])['WScore'].agg('sum'))
a['id1'] = a.index
a['id1'] = a['id1'].apply(lambda x: x[0])

b = pd.DataFrame(MRegularSeasonDetailedResults.loc[(MRegularSeasonDetailedResults.WTeamID == MRegularSeasonDetailedResults.min_id) ,['id','WTeamID','Season']].groupby(['id','Season'])['WTeamID'].agg('count'))
b['id1'] = b.index
b['id1'] = b['id1'].apply(lambda x: x[0])

match_agg= pd.merge(a,b,left_on='id1',right_on='id1',how='left')

a= pd.DataFrame(MRegularSeasonDetailedResults.loc[(MRegularSeasonDetailedResults.LTeamID == MRegularSeasonDetailedResults.min_id) ,['id','LTeamID','Season']].groupby(['id','Season'])['LTeamID'].agg('count'))
a['id1'] = a.index
a['id1'] = a['id1'].apply(lambda x: x[0])
match_agg = pd.merge(match_agg,a,left_on = 'id1',right_on = 'id1',how = 'left')

a = pd.DataFrame(MRegularSeasonDetailedResults.loc[(MRegularSeasonDetailedResults.LTeamID == MRegularSeasonDetailedResults.min_id ) ,['id','LScore','Season']].groupby(['id','Season'])['LScore'].agg('sum'))
a['id1'] = a.index
a['id1'] = a['id1'].apply(lambda x: x[0])
match_agg = pd.merge(match_agg,a,left_on = 'id1',right_on = 'id1',how = 'left')
match_agg = match_agg.rename(columns = {'WScore' : 'T1WScore','LScore' : 'T1LScore',
                              'WTeamID' : 'Wins','LTeamID' : 'Losses'})

a = pd.DataFrame(MRegularSeasonDetailedResults.loc[(MRegularSeasonDetailedResults.WTeamID == MRegularSeasonDetailedResults.max_id ) ,['id','WScore','Season']].groupby(['id','Season'])['WScore'].agg('sum'))
a['id1'] = a.index
a['id1'] = a['id1'].apply(lambda x: x[0])
match_agg = pd.merge(match_agg,a,left_on = 'id1',right_on = 'id1',how = 'left')

a = pd.DataFrame(MRegularSeasonDetailedResults.loc[(MRegularSeasonDetailedResults.LTeamID == MRegularSeasonDetailedResults.max_id ) ,['id','LScore','Season']].groupby(['id','Season'])['LScore'].agg('sum'))
a['id1'] = a.index
a['id1'] = a['id1'].apply(lambda x: x[0])
match_agg = pd.merge(match_agg,a,left_on = 'id1',right_on = 'id1',how = 'left')
match_agg = match_agg.rename(columns = {'WScore' : 'T2WScore','LScore' : 'T2LScore'})       
match_agg.fillna(0,inplace = True)
match_agg = pd.merge(match_agg,MRegularSeasonDetailedResults[['id','NumOT','min_id','max_id','Season']],left_on = 'id1',right_on = 'id',how = 'left').drop(['id1'], axis=1)

match_agg['min_id1'] = match_agg.Season.map(str) + '_' + match_agg.min_id.map(str)
match_agg['max_id1'] = match_agg.Season.map(str) + '_' + match_agg.max_id.map(str)


match_agg = pd.merge(match_agg,team_agg[['id_1','Avg_Win_Points','Avg_Lose_Points','Avg_Points','Matches_Played','Win_perc','Lose_perc']],left_on = 'min_id1',right_on = 'id_1',how = 'left').drop(['id_1','min_id1'], axis=1)
match_agg = match_agg.rename(columns = {'Avg_Win_Points' : 'T1Avg_Win_Points','Avg_Lose_Points': 'T1Avg_Lose_Points',
                                        'Avg_Points': 'T1Avg_Points','Matches_Played': 'T1Matches_Played','Win_perc': 'T1Win_perc','Lose_perc': 'T1Lose_perc'})
match_agg = pd.merge(match_agg,team_agg[['id_1','Avg_Win_Points','Avg_Lose_Points','Avg_Points','Matches_Played','Win_perc','Lose_perc']],left_on = 'max_id1',right_on = 'id_1',how = 'left').drop(['id_1','max_id1'], axis=1)
match_agg = match_agg.rename(columns = {'Avg_Win_Points' : 'T2Avg_Win_Points','Avg_Lose_Points': 'T2Avg_Lose_Points',
                                        'Avg_Points': 'T2Avg_Points','Matches_Played': 'T2Matches_Played','Win_perc': 'T2Win_perc','Lose_perc': 'T2Lose_perc'})
match_agg.drop_duplicates(subset = 'id',inplace = True)

match_agg['Avg_Points_Scored'] = (match_agg.T1WScore + match_agg.T1LScore)/(match_agg.Wins + match_agg.Losses)
match_agg['Avg_Points_Conceded'] = (match_agg.T2WScore + match_agg.T2LScore)/(match_agg.Wins + match_agg.Losses)
match_agg.drop(['T1WScore','T2WScore','T1LScore','T2LScore','T1Lose_perc','T2Lose_perc'],inplace = True,axis = 1)

match_agg = pd.merge(match_agg,MNCAATourneyCompactResults[['Event','id']],left_on = 'id',right_on = 'id',how = 'left')
match_agg.groupby('Event')['Event'].agg('count')
