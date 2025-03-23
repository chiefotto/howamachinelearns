import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import time
from nba_api.stats.endpoints import teamgamelogs, playergamelogs,leaguedashteamstats
from nba_api.stats.static import teams, players
from nba_api.live.nba.endpoints import scoreboard
# ========== CACHED FUNCTIONS ==========

@st.cache_data(show_spinner=False)
def get_all_nba_players():
    return players.get_players()

@st.cache_data(show_spinner=False)
def get_all_nba_teams():
    return teams.get_teams()


@st.cache_data(show_spinner=False)
def get_todays_games_cleaned():

    games = scoreboard.ScoreBoard()
    data = games.get_dict()
    game_list = data.get("scoreboard", {}).get("games", [])

    simplified_games = []

    for game in game_list:
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})

        simplified_games.append({
            "gameId": game.get("gameId"),
            "home": {
                "teamId": home.get("teamId"),
                "teamName": home.get("teamName"),
                "teamCity": home.get("teamCity"),
                "teamTricode": home.get("teamTricode"),
            },
            "away": {
                "teamId": away.get("teamId"),
                "teamName": away.get("teamName"),
                "teamCity": away.get("teamCity"),
                "teamTricode": away.get("teamTricode"),
            }
        })

    return simplified_games



@st.cache_data
def get_all_teams():
    team_list = teams.get_teams()
    # Make a dict of full name (display) to ID (value)
    return {f"{team['full_name']} ({team['abbreviation']})": team["id"] for team in team_list}
    


@st.cache_data(show_spinner=True)
def fetch_nba_last_25():
    log = teamgamelogs.TeamGameLogs(season_nullable="2024-25", last_n_games_nullable=25)
    return log.get_data_frames()[0]

@st.cache_data
def get_team_id_name_map():
    return {team['id']: team['full_name'] for team in teams.get_teams()}


# ========== HELPERS ==========

def preprocess_data(df):
    df = df[['TEAM_NAME','MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA','FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA','PF', 'PFD', 'PTS', 'PLUS_MINUS']]
    df_agg = df.groupby('TEAM_NAME').mean().reset_index()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_agg.drop(columns=["TEAM_NAME"]))
    df_scaled = pd.DataFrame(df_scaled, columns=df_agg.columns[1:])
    df_scaled['TEAM_NAME'] = df_agg['TEAM_NAME']
    return df_scaled

def elbow_method(df):
    silhouette_scores = []
    df_no_team_name = df.drop(columns=['TEAM_NAME'])
    k_range = range(2, 12)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_no_team_name)
        score = silhouette_score(df_no_team_name, labels)
        silhouette_scores.append(score)
    fig, ax = plt.subplots()
    ax.plot(k_range, silhouette_scores, marker='o')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal K Using Silhouette Score")
    return fig

def kmeans_cal(df, n):
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df.drop(columns=['TEAM_NAME']))
    return df

def generate_cluster_team_ids(df):
    nba_teams = get_all_nba_teams()
    name_to_id = {team['full_name']: team['id'] for team in nba_teams}
    cluster_team_ids = {}
    for cluster in df['Cluster'].unique():
        cluster_name = f"cluster_{cluster}"
        team_names = df[df['Cluster'] == cluster]['TEAM_NAME']
        team_ids = [name_to_id[name] for name in team_names if name in name_to_id]
        cluster_team_ids[cluster_name] = team_ids
    return cluster_team_ids

def generate_team_ids(selected_teams):
    nba_teams = get_all_nba_teams()
    name_to_id = {team['full_name']: team['id'] for team in nba_teams}
    return name_to_id
    

def get_player_ids_from_input(player_input):
    nba_players = get_all_nba_players()
    name_to_id = {player["full_name"]: player["id"] for player in nba_players}
    player_names = [name.strip() for name in player_input.split(",")]
    return [name_to_id[name] for name in player_names if name in name_to_id]

# def view_clusters(scaled_df):
#     df = kmeans_cal(scaled_df, 6)
#     sorted_df = df[['TEAM_NAME', 'Cluster']].sort_values(by='Cluster')
#     cluster_team_ids = generate_cluster_team_ids(sorted_df)
#     st.session_state["cluster_team_ids"] = cluster_team_ids
#     st.session_state["cluster_df"] = sorted_df
#     st.write("### Clustered Teams")
#     st.dataframe(sorted_df)

# def fetch_players_vs_cluster(player_ids, cluster_teams_dict, season="2024-25", rate_limit=3):
#     for player_id in player_ids:
#         for cluster, team_ids in cluster_teams_dict.items():
#             player_vs_cluster = pd.DataFrame()
#             for team_id in team_ids:
#                 try:
#                     logs = playergamelogs.PlayerGameLogs(
#                         season_nullable=season,
#                         player_id_nullable=player_id,
#                         opp_team_id_nullable=team_id
#                     ).get_data_frames()[0]
#                     if logs.empty:
#                         continue
#                     player_vs_cluster = pd.concat([player_vs_cluster, logs], ignore_index=True)
#                     time.sleep(rate_limit)
#                 except Exception as e:
#                     st.warning(f"Error fetching data for Player {player_id} vs. Team {team_id}: {e}")
#             if not player_vs_cluster.empty:
#                 st.write(f"### Player ID: {player_id} vs. {cluster}")
#                 st.dataframe(player_vs_cluster)

def fetch_players_vs_cluster(player_ids, opponent_team_ids, season="2024-25", rate_limit=3):
    for player_id in player_ids:
        player_vs_opponents = pd.DataFrame()

        for opponent_id in opponent_team_ids:
            try:
                logs = playergamelogs.PlayerGameLogs(
                    season_nullable=season,
                    player_id_nullable=player_id,
                    opp_team_id_nullable=opponent_id
                ).get_data_frames()[0]

                if logs.empty:
                    continue

                player_vs_opponents = pd.concat([player_vs_opponents, logs], ignore_index=True)
                time.sleep(rate_limit)
            except Exception as e:
                st.warning(f"Error fetching stats: Player {player_id} vs Team {opponent_id} - {e}")

        if not player_vs_opponents.empty:
            st.write(f"### Player ID {player_id} vs Selected Cluster")
            st.dataframe(player_vs_opponents)



# def fetch_team_vs_team (selected_teams,cluster_teams_dict, season="2024-25", rate_limit=3):
#     for selected_team in selected_teams:
#         for cluster, team_ids in cluster_teams_dict.items():
#             team_vs_cluster = pd.DataFrame()
#             for team_id in team_ids:
#                 try:
#                     logs = teamgamelogs.TeamGameLogs(season_nullable=season, team_id_nullable=selected_team, opp_team_id_nullable=team_id).get_data_frames()[0]
#                     if logs.empty:
#                         continue
#                     team_vs_cluster = pd.concat([team_vs_cluster,logs], ignore_index=True)
#                     time.sleep(rate_limit)
#                 except Exception as e:
#                     st.warning(f"error fetching games vs {cluster}")
#             if not team_vs_cluster.empty:
#                     st.write(f"### Player ID: {selected_team} vs. {cluster}")
#                     st.dataframe(team_vs_cluster)


def get_team_vs_each_cluster_team(selected_team_ids, opponent_team_ids, season="2024-25", rate_limit=1.5):
    all_data = []

    for team_id in selected_team_ids:
        for opp_id in opponent_team_ids:
            try:
                result = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    season_type_all_star="Regular Season",
                    team_id_nullable=team_id,
                    opponent_team_id=opp_id
                )
                df = result.get_data_frames()[0]
                if not df.empty:
                    df["OPPONENT_TEAM_ID"] = opp_id  # Optionally label opponent
                    all_data.append(df)
                time.sleep(rate_limit)
            except Exception as e:
                st.warning(f"Error fetching Team {team_id} vs Opponent {opp_id}: {e}")

    # ✅ Return one single merged DataFrame
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        # 👇 Add readable opponent name
        team_id_to_name = get_team_id_name_map()
        final_df["OPPONENT_TEAM_NAME"] = final_df["OPPONENT_TEAM_ID"].map(team_id_to_name)

        return final_df
    else:
        return pd.DataFrame()

            


    
    

# ========== STREAMLIT UI ==========

# Streamlit UI
st.title("🏀 NBA Cluster Tool — Jokr")
st.subheader("Step 1: Fetch & Analyze Data")

raw_df = fetch_nba_last_25()
processed_df = preprocess_data(raw_df)
games = get_todays_games_cleaned()
team_map = get_all_teams()


print(games)

for game in games:
    st.write(
        f"Away: {game['away']['teamTricode']} ({game['away']['teamCity']}) "
        f"@ {game['home']['teamTricode']} ({game['home']['teamCity']}) :Home"
    )



# Show silhouette score graph
if st.button("📈 Display Optimal K Graph"):
    st.pyplot(elbow_method(processed_df))

# Step 2: User selects K
k_val = st.number_input("Select number of clusters (k)", min_value=2, max_value=10, value=6, step=1)

# Step 3: Generate clusters
if st.button("🔁 Generate & View Clusters"):
    df_clustered = kmeans_cal(processed_df.copy(), k_val)
    cluster_team_ids = generate_cluster_team_ids(df_clustered)
    st.session_state["cluster_team_ids"] = cluster_team_ids
    st.session_state["cluster_df"] = df_clustered[['TEAM_NAME', 'Cluster']].sort_values(by='Cluster')

# Always show clustered team table if available
if "cluster_df" in st.session_state:
    st.subheader(" Clustered Teams Table")
    st.dataframe(st.session_state["cluster_df"])

# Step 4: Let user select clusters and players
if "cluster_team_ids" in st.session_state:
    available_clusters = list(st.session_state["cluster_team_ids"].keys())
    selected_clusters = st.multiselect("Select Clusters to Analyze", available_clusters)
else:
    selected_clusters = []
    

cluster_teams_dict = st.session_state.get("cluster_team_ids", {})
selected_cluster_teams = {
            cluster: cluster_teams_dict[cluster] for cluster in selected_clusters
        }
flat_team_ids = []
for team_list in selected_cluster_teams.values():
    flat_team_ids.extend(team_list)

    

selected_team_names = st.multiselect(
    "Select one or more NBA teams:",
    options=list(team_map.keys())  # team display names
)

selected_team_ids = [team_map[name] for name in selected_team_names]





if st.button("📊 Show Raw Stats vs Each Team in Cluster"):
    if selected_team_ids and flat_team_ids:
        df = get_team_vs_each_cluster_team(selected_team_ids, flat_team_ids)
        if not df.empty:
            team_id_to_name = get_team_id_name_map()
            for team_id in selected_team_ids:
                team_df = df[df["TEAM_ID"] == team_id]
                team_name = team_id_to_name.get(team_id, f"Team {team_id}")
                
                st.write(f"### {team_name} vs Selected Cluster")
                st.dataframe(team_df)
        else:
            st.info("No data found.")

        

player_input = st.text_input("Enter Player Names (comma-separated & case-sensitive):")

# Step 5: Fetch and show player data
if st.button("📊 Fetch Player vs Cluster Stats"):
    if player_input and selected_clusters:
        player_ids = get_player_ids_from_input(player_input)
        fetch_players_vs_cluster(player_ids, flat_team_ids)
    else:
        st.warning("Please enter players and select clusters first.")
        
    
        
print(flat_team_ids, "flat team ids")

