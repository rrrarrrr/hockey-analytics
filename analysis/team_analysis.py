# src/analysis/team_analysis.py
"""
Анализ по командам: общий пробег, скорость, активность в зонах.
"""
import os
import json
import pandas as pd

# --- Пути ---
TRACKS_FOLDER = r"C:/opencv/env/data/tracks"
TEAM_STATS_PATH = r"C:/opencv/env/data/team_stats.xlsx"

# --- Параметры ---
FRAME_WIDTH = 784
FPS = 30

# --- Статистика ---
team_distance = {"blue": 0.0, "yellow": 0.0}
team_max_speed = {"blue": 0.0, "yellow": 0.0}
team_zone_stats = {
    "blue": {"defensive": 0, "neutral": 0, "offensive": 0},
    "yellow": {"defensive": 0, "neutral": 0, "offensive": 0}
}

def get_zone(x):
    if x < FRAME_WIDTH * 0.33:
        return "defensive"
    elif x < FRAME_WIDTH * 0.66:
        return "neutral"
    else:
        return "offensive"

def main():
    print("We begin the analysis by teams...")

    if not os.path.exists(TRACKS_FOLDER):
        print(f" Folder not found: {TRACKS_FOLDER}")
        return

    frame_files = sorted([f for f in os.listdir(TRACKS_FOLDER) if f.endswith('.json')])

    for file in frame_files:
        with open(os.path.join(TRACKS_FOLDER, file), 'r') as f:
            try:
                data = json.load(f)
            except:
                continue

        for player in data.get("players", []):
            team = player.get("team_color", "unknown")
            if team not in ["blue", "yellow"]:
                continue

            x = (player["bbox"][0] + player["bbox"][2]) / 2
            zone = get_zone(x)

            team_distance[team] += player.get("distance_m", 0.0)
            speed = player.get("speed_m_s", 0.0)
            if speed > team_max_speed[team]:
                team_max_speed[team] = speed

            team_zone_stats[team][zone] += 1

    # Подготовка данных
    team_data = []
    for team, name in [("blue", "Blue"), ("yellow", "Yellow")]:
        team_data.append({
            "Team": name,
            "Total Distance (m)": round(team_distance[team], 2),
            "Max Speed (m/s)": round(team_max_speed[team], 2),
            "Defensive Zone Visits": team_zone_stats[team]["defensive"],
            "Neutral Zone Visits": team_zone_stats[team]["neutral"],
            "Offensive Zone Visits": team_zone_stats[team]["offensive"]
        })

    # Сохранение
    df = pd.DataFrame(team_data)
    df.to_excel(TEAM_STATS_PATH, index=False)
    print(f"Team statistics are saved: {TEAM_STATS_PATH}")

if __name__ == "__main__":
    main()
