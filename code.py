# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:33:08 2020

@author: Shubham Kumar
"""
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
MRegularSeasonDetailedResults['min_id'] = MRegularSeasonDetailedResults.Season.map(str) + '_' + MRegularSeasonDetailedResults.WTeamID.map(str)
MRegularSeasonDetailedResults['max_id'] = MRegularSeasonDetailedResults.Season.map(str) + '_' + MRegularSeasonDetailedResults.LTeamID.map(str)

MNCAATourneyDetailedResults['id'] = MNCAATourneyDetailedResults.Season.map(str) + '_' + MNCAATourneyDetailedResults[['WTeamID','LTeamID']].min(axis = 1).map(str) + '_' +MNCAATourneyDetailedResults[['WTeamID','LTeamID']].max(axis = 1).map(str)
MNCAATourneyDetailedResults['min_id'] = MNCAATourneyDetailedResults.Season.map(str) + '_' + MNCAATourneyDetailedResults[['WTeamID','LTeamID']].min(axis = 1).map(str)
MNCAATourneyDetailedResults['max_id'] = MNCAATourneyDetailedResults.Season.map(str) + '_' + MNCAATourneyDetailedResults[['WTeamID','LTeamID']].max(axis = 1).map(str)

Teams = pd.DataFrame()
Teams = pd.DataFrame(pd.concat([MNCAATourneyDetailedResults['min_id'],MNCAATourneyDetailedResults['max_id']]).unique())
Teams = Teams.rename(columns = {0 : 'id'})  

a = pd.DataFrame(MRegularSeasonDetailedResults.groupby(['min_id'])['WScore','WFGM','WFGA',
                 'WFGM3','WFTM', 'WFTA', 'WOR', 'WDR','WAst', 'WTO', 'WStl', 'WBlk', 'WPF'].agg('sum'))
a['Wins'] = pd.DataFrame(MRegularSeasonDetailedResults.groupby(['min_id'])['WTeamID'].agg('count'))
b = pd.DataFrame(MRegularSeasonDetailedResults.groupby(['max_id'])['LScore','NumOT','LFGM','LFGA',
                 'LFGM3','LFTM', 'LFTA', 'LOR', 'LDR','LAst', 'LTO', 'LStl', 'LBlk', 'LPF'].agg('sum'))
b['Losses'] = pd.DataFrame(MRegularSeasonDetailedResults.groupby(['max_id'])['LTeamID'].agg('count'))

ab = pd.merge(a,b,how = 'outer',right_index = True,left_index = True)

ab.fillna(0,inplace = True)

Teams = pd.merge(Teams,ab,how = 'left',left_on = 'id',right_index = True)

Teams['AvgPoints'] = (Teams.WScore + Teams.LScore)/(Teams.Wins + Teams.Losses)
Teams['AvgFGM'] = (Teams.WFGM + Teams.LFGM)/(Teams.Wins + Teams.Losses)
Teams['AvgFGM3'] = (Teams.WFGM3 + Teams.LFGM3)/(Teams.Wins + Teams.Losses)
Teams['AvgFTM'] = (Teams.WFTM + Teams.LFTM)/(Teams.Wins + Teams.Losses)
Teams['AvgFTA'] = (Teams.WFTA + Teams.LFTA)/(Teams.Wins + Teams.Losses)
Teams['AvgOR'] = (Teams.WOR + Teams.LOR)/(Teams.Wins + Teams.Losses)
Teams['AvgDR'] = (Teams.WDR + Teams.LDR)/(Teams.Wins + Teams.Losses)
Teams['AvgAst'] = (Teams.WAst + Teams.LAst)/(Teams.Wins + Teams.Losses)
Teams['AvgTO'] = (Teams.WTO + Teams.LTO)/(Teams.Wins + Teams.Losses)
Teams['AvgStl'] = (Teams.WStl + Teams.LStl)/(Teams.Wins + Teams.Losses)
Teams['AvgBlk'] = (Teams.WBlk + Teams.LBlk)/(Teams.Wins + Teams.Losses)
Teams['AvgPF'] = (Teams.WPF + Teams.LPF)/(Teams.Wins + Teams.Losses)

Teams.drop(['WScore','WFGM','WFGA','LScore','LFGM','LFGA',
                 'LFGM3','LFTM', 'LFTA', 'LOR', 'LDR','LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
                 'WFGM3','WFTM', 'WFTA', 'WOR', 'WDR','WAst', 'WTO', 'WStl', 'WBlk', 'WPF'],axis = 1,inplace = True)
Teams.columns = 'T1' + Teams.columns
import numpy as np
MNCAATourneyCompactResults = MNCAATourneyCompactResults[MNCAATourneyCompactResults.Season > 2002]
MNCAATourneyCompactResults['id'] = MNCAATourneyCompactResults.Season.map(str) + '_' + MNCAATourneyCompactResults[['WTeamID','LTeamID']].min(axis = 1).map(str) + '_' +MNCAATourneyCompactResults[['WTeamID','LTeamID']].max(axis = 1).map(str)
MNCAATourneyCompactResults['min_id'] = MNCAATourneyCompactResults.Season.map(str) + '_' + MNCAATourneyCompactResults[['WTeamID','LTeamID']].min(axis = 1).map(str)
MNCAATourneyCompactResults['max_id'] = MNCAATourneyCompactResults.Season.map(str) + '_' + MNCAATourneyCompactResults[['WTeamID','LTeamID']].max(axis = 1).map(str)

MNCAATourneyCompactResults['Event'] = np.where(MNCAATourneyCompactResults.min_id == MNCAATourneyCompactResults.WTeamID,1,0)

data = pd.merge(MNCAATourneyCompactResults[['id','min_id','max_id','Event']],Teams,how = 'left', left_on = 'min_id',
                right_on = 'T1id')
Teams.columns = Teams.columns.str.replace('T1', 'T2')
data = pd.merge(data,Teams,how = 'left', left_on = 'max_id',
                right_on = 'T2id')
data.drop(['min_id', 'max_id','T1id','T2id'],axis = 1,inplace = True)
