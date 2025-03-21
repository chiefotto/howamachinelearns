import time
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
from nba_api.stats.static import teams
from nba_api.stats.static import players
from sklearn.cluster import KMeans
from nba_api.stats.endpoints import playergamelogs
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

    
    
def view_clusters(scaled_df):
    """
    Generates clusters and stores the cluster-to-team ID mappings.
    """
    df = kmeans_cal(scaled_df, 6)  # Perform clustering
    sorted_df = df[['TEAM_NAME', 'Cluster']].sort_values(by='Cluster', ascending=True)

    # Generate dynamic team mappings
    cluster_team_ids = generate_cluster_team_ids(sorted_df) 

    # Store globally so it's available for later API requests
    st.session_state["cluster_team_ids"] = cluster_team_ids  

    # Display sorted cluster list
    st.write("### Clustered Teams")
    st.dataframe(sorted_df)  # Display the sorted teams with clusters




def generate_cluster_team_ids(df):
    """
    Given a DataFrame with 'TEAM_NAME' and 'Cluster', this function dynamically creates
    lists of team IDs for each cluster, storing them with variable names like 'cluster_0', 'cluster_1', etc.

    :param df: DataFrame containing 'TEAM_NAME' and 'Cluster'
    :return: Dictionary where keys are cluster names (e.g., 'cluster_0') and values are lists of team IDs
    """
    cluster_team_ids = {}  # Dictionary to store team IDs for each cluster

    for cluster in df['Cluster'].unique():  # Iterate through each unique cluster
        cluster_teams = df[df['Cluster'] == cluster]['TEAM_NAME']  # Get teams in the cluster
        
        team_ids = []  # List to store team IDs for this cluster
        for team_name in cluster_teams:
            team_data = teams.find_teams_by_full_name(team_name)  # Get team info
            if team_data:  # Ensure the team exists
                team_ids.append(team_data[0]['id'])  # Append the first matching team ID
        
        # Dynamically store the list under a formatted cluster name
        cluster_name = f"cluster_{cluster}"
        cluster_team_ids[cluster_name] = team_ids  # Store in dictionary
    
    return cluster_team_ids


def get_player_ids_from_input(player_input):
    """
    Given a comma-separated string of player names, return a list of player IDs.
    
    :param player_input: String of player names (e.g., "LeBron James, Kevin Durant, Stephen Curry")
    :return: List of player IDs
    """
    # Get all NBA players
    nba_players = players.get_players()

    # Create a mapping of full names to player IDs
    name_to_id = {player["full_name"]: player["id"] for player in nba_players}

    # Split input by commas, strip spaces
    player_names = [name.strip() for name in player_input.split(",")]

    # Get player IDs
    player_ids = [name_to_id[name] for name in player_names if name in name_to_id]

    return player_ids



def fetch_players_vs_cluster(player_ids, cluster_teams_dict, season="2024-25", rate_limit=3):
    """
    Fetches player performance vs. selected cluster teams dynamically.

    :param player_ids: List of player IDs
    :param cluster_teams_dict: Dictionary where keys are cluster names (e.g., 'cluster_0') and values are lists of team IDs
    :param season: NBA season (default "2024-25")
    :param rate_limit: Time (seconds) to wait between API calls
    """
    for player_id in player_ids:
        for cluster, team_ids in cluster_teams_dict.items():
            player_vs_cluster = pd.DataFrame()

            for team_id in team_ids:
                try:
                    # Fetch game logs for the player against the specific team
                    game_logs = playergamelogs.PlayerGameLogs(
                        season_nullable=season,
                        player_id_nullable=player_id,
                        opp_team_id_nullable=team_id
                    ).get_data_frames()[0]

                    # Skip if no data found
                    if game_logs.empty:
                        continue

                    # Append data
                    player_vs_cluster = pd.concat([player_vs_cluster, game_logs], ignore_index=True)

                    # Sleep to avoid API rate limiting
                    time.sleep(rate_limit)

                except Exception as e:
                    st.write(f"Error fetching data for Player {player_id} vs. Team {team_id}: {e}")

            # Display each player's DataFrame per cluster
            if not player_vs_cluster.empty:
                st.write(f"### Player ID: {player_id} vs. {cluster}")
                st.dataframe(player_vs_cluster)

    
    
raw_df = fetch_nba_last_25()
processed_df = preprocess_data(raw_df)

st.title("NBA Cluster Tool -- Jokr")

st.subheader("When using this tool, calculate clusters to or skip, then input")

if st.button("Fetch and Display Optimal K graph"):
        with st.spinner("Fetching..."):
            
            optimal_k = elbow_method(processed_df)
            st.pyplot(optimal_k)


st.subheader("once the graphs done generating insert cluster amount and run")

if st.button("Generate and View Clusters"):
    view_clusters(processed_df)  # This will now store cluster_team_ids in session_state





options = st.multiselect("choose clsuters",["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5"])


player_input = st.text_input("Insert players (comma-separated) and CASE SENSITIVE", "")
if player_input:
    player_ids = get_player_ids_from_input(player_input)
    st.write("Player IDs:", player_ids)



if st.button("Fetch Player vs Cluster Data"):
    if player_input and options:
        player_ids = get_player_ids_from_input(player_input)

        # Ensure clusters are generated first
        if "cluster_team_ids" in st.session_state:
            cluster_teams_dict = st.session_state["cluster_team_ids"]

            # Filter only selected clusters
            selected_cluster_teams = {cluster: cluster_teams_dict[cluster] for cluster in options if cluster in cluster_teams_dict}

            fetch_players_vs_cluster(player_ids, selected_cluster_teams)

        else:
            st.write("⚠️ Please generate clusters first before fetching player data.")
    else:
        st.write("⚠️ Please enter player names and select at least one cluster.")

        







