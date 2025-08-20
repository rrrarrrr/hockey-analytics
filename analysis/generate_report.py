# src/analysis/generate_report.py
"""
Генерация HTML-отчёта с диаграммами.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Пути ---
TEAM_STATS_PATH = r"C:/opencv/env/data/team_stats.xlsx"
HTML_REPORT_PATH = r"C:/opencv/env/data/hockey_report.html"
IMAGES_FOLDER = r"C:/opencv/env/data/report_images"
os.makedirs(IMAGES_FOLDER, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_distance_chart(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df['Team'], df['Total Distance (m)'], color=['blue', 'orange'], alpha=0.7)
    plt.title('Total Distance by Team')
    plt.ylabel('Distance (m)')
    for i, v in enumerate(df['Total Distance (m)']):
        plt.text(i, v + 10, f"{v:.0f} m", ha='center')
    plt.tight_layout()
    path = os.path.join(IMAGES_FOLDER, "distance_chart.png")
    plt.savefig(path)
    plt.close()
    return "report_images/distance_chart.png"

def create_zone_chart(df):
    defensive = df['Defensive Zone Visits'].sum()
    neutral = df['Neutral Zone Visits'].sum()
    offensive = df['Offensive Zone Visits'].sum()
    total = defensive + neutral + offensive

    if total == 0:
        return None

    sizes = [defensive, neutral, offensive]
    labels = [f'Defensive ({defensive})', f'Neutral ({neutral})', f'Offensive ({offensive})']
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Zone Activity Distribution')
    plt.tight_layout()
    path = os.path.join(IMAGES_FOLDER, "zone_chart.png")
    plt.savefig(path)
    plt.close()
    return "report_images/zone_chart.png"
def create_distance_chart(df):
    """Гистограмма: общий пробег по командам."""
    if df.empty:
        return None
    plt.figure(figsize=(8, 5))
    plt.bar(df['Team'], df['Total Distance (m)'], color=['blue', 'orange'], alpha=0.7)
    plt.title('Total Distance by Team')
    plt.ylabel('Distance (m)')
    for i, v in enumerate(df['Total Distance (m)']):
        plt.text(i, v + 10, f"{v:.0f} m", ha='center')
    plt.tight_layout()
    path = os.path.join(IMAGES_FOLDER, "distance_chart.png")
    plt.savefig(path)
    plt.close()
    return "report_images/distance_chart.png"

def create_zone_chart(df):
    """Круговая диаграмма: активность в зонах."""
    if df.empty:
        return None
    defensive = df['Defensive Zone Visits'].sum()
    neutral = df['Neutral Zone Visits'].sum()
    offensive = df['Offensive Zone Visits'].sum()
    total = defensive + neutral + offensive
    if total == 0:
        return None
    sizes = [defensive, neutral, offensive]
    labels = [f'Defensive ({defensive})', f'Neutral ({neutral})', f'Offensive ({offensive})']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Zone Activity Distribution')
    plt.tight_layout()
    path = os.path.join(IMAGES_FOLDER, "zone_chart.png")
    plt.savefig(path)
    plt.close()
    return "report_images/zone_chart.png"

def generate_html_report(df):
    dist_img = create_distance_chart(df)
    zone_img = create_zone_chart(df)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Hockey Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1a3e6a; }}
            img {{ max-width: 100%; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Hockey Match Analysis Report</h1>
        <h2>Total Distance by Team</h2>
        <img src="{dist_img}" alt="Distance">
        <h2>Zone Activity</h2>
        <img src="{zone_img}" alt="Zones">
    </body>
    </html>
    """
    with open(HTML_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"The HTML report is saved: {HTML_REPORT_PATH}")

def main():
    print("Generating a report...")
    if not os.path.exists(TEAM_STATS_PATH):
        print(f"File not found:{TEAM_STATS_PATH}")
        return

    df = pd.read_excel(TEAM_STATS_PATH)
    generate_html_report(df)

if __name__ == "__main__":
    main()
