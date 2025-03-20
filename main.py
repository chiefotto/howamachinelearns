from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import pandas as pd 
import numpy as np 
from nba_api.stats.endpoints import LeagueGameLog
from nba_api.stats.endpoints import teamgamelogs
from sklearn.cluster import KMeans
import datetime
import os


### fetching nba data
def fetch_nba_last_25 ():
    log = teamgamelogs.TeamGameLogs(season_nullable="2024-25", last_n_games_nullable=25)
    df = log.get_data_frames()[0]
    return df


def preprocess_data (df):
        df = df = df[['TEAM_NAME','MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA',
       'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA',
       'PF', 'PFD', 'PTS', 'PLUS_MINUS']]
        df_agg = df.groupby('TEAM_NAME').mean().reset_index()
        ### returns teams data aggregated df 

        scaler = StandardScaler()
        df = scaler.fit_transform(df_agg.drop(columns=["TEAM_NAME"]))
        df = pd.DataFrame(df,columns = df_agg.columns[1:])
        df['TEAM_NAME'] = df_agg['TEAM_NAME']
        return df 

def elbow_method(df):
    silhouette_scores = []
    df_no_team_name = df.drop(columns=['TEAM_NAME'])
    k_range = range(2,12)
    for k in k_range :
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_no_team_name)
        score = silhouette_score(df_no_team_name,labels)
        silhouette_scores.append(score)
    fig, ax = plt.subplots()
    ax.plot(k_range, silhouette_scores, marker='o')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal K Using Silhouette Score")
    return fig

st.title("Grab the last 25 NBA Games data and find the optimal K")

if st.button("Fetch and Display Optimal K graph"):
        with st.spinner("Fetching..."):
            raw_df = fetch_nba_last_25()
            processed_df = preprocess_data(raw_df)
            optimal_k = elbow_method(processed_df)
            st.pyplot(optimal_k)




    
        







