from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
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


def kmeans_cal(df,n):
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df.drop(columns = ['TEAM_NAME']))
    return df


def pca_calc(df):
    pca = PCA(n_components=3)
    df_scaled_pca = pca.fit_transform((kmeans_cal(df, 6)).drop(columns=['TEAM_NAME','Cluster']))
    return df_scaled_pca
    
def plot_clusters (df):
    computed_clusters = pca_calc(df)
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=computed_clusters, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
    plt.title("NBA Team Clusters (Last 25 Games, k=6)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    for i, row in computed_clusters.iterrows():
        plt.text(row['PC1'], row['PC2'], row['TEAM_NAME'], fontsize=8, ha='left')
    plt.show()
    
    
def view_clusters (scaled_df):
    df = kmeans_cal(scaled_df,6)
    sorted_df = df[['TEAM_NAME','Cluster']].sort_values(by='Cluster', ascending = True)
    sorted_df
    
    
    
    
    
    
    
    
    
    
    
    
raw_df = fetch_nba_last_25()
processed_df = preprocess_data(raw_df)

st.title("NBA Cluster Tool -- Jokr")

st.subheader("When using this tool, calculate clusters to or skip, then input")

if st.button("Fetch and Display Optimal K graph"):
        with st.spinner("Fetching..."):
            
            optimal_k = elbow_method(processed_df)
            st.pyplot(optimal_k)


st.subheader("once the graphs done generating insert cluster amount and run")

st.button("generate and view clusters", on_click=view_clusters(processed_df))




options = st.multiselect("choose clsuters",["cluster 0", "cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5"])


player  = st.text_input("insert players")
    
        







