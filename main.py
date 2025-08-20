# main.py
"""
Полный пайплайн анализа хоккея: извлечение → трекинг → визуализация → отчёт.
С распознаванием номеров и детекцией шайбы.
"""

import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv
import easyocr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Пути ---
VIDEO_PATH = r"C:/opencv/env/data/raw/hockey_match.mp4"
FRAMES_FOLDER = r"C:/opencv/env/data/frames"
TRACKS_FOLDER = r"C:/opencv/env/data/tracks"
TRACKED_FOLDER = r"C:/opencv/env/data/tracked"
OUTPUT_VIDEO = r"C:/opencv/env/data/hockey_analysis.mp4"
STATS_JSON = r"C:/opencv/env/data/stats.json"
EXCEL_PATH = r"C:/opencv/env/data/stats.xlsx"
TEAM_STATS_PATH = r"C:/opencv/env/data/team_stats.xlsx"
HTML_REPORT_PATH = r"C:/opencv/env/data/hockey_report.html"
IMAGES_FOLDER = r"C:/opencv/env/data/report_images"

# --- Параметры ---
SAVE_EVERY = 10
FPS = 30
FRAME_WIDTH = 784
FRAME_HEIGHT = 442

# --- Цвета ---
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3), dtype=int)

# --- Глобальные переменные ---
player_positions = {}
player_distances = {}
player_speeds = {}
PIXEL_TO_METER = 0.05

# --- Загрузка модели и OCR ---
print("Loading YOLOv8...")
model_players = YOLO('yolov8n.pt')
model_puck = YOLO(r"C:\opencv\runs\detect\puck_detection_retrained2\weights\best.pt")
print("Loading the OCR...")
reader = easyocr.Reader(['en'], gpu=False)
tracker = sv.ByteTrack()

# --- Функции ---

def extract_jersey_number(frame, x1, y1, x2, y2):
    """Распознаёт номер на майке."""
    h = y2 - y1
    w = x2 - x1
    jersey_y1 = int(y1 + 0.6 * h)
    jersey_y2 = int(y2)
    jersey_x1 = int(x1 + 0.1 * w)
    jersey_x2 = int(x2 - 0.1 * w)
    jersey_y1 = max(jersey_y1, 0)
    jersey_y2 = min(jersey_y2, frame.shape[0])
    jersey_x1 = max(jersey_x1, 0)
    jersey_x2 = min(jersey_x2, frame.shape[1])
    if jersey_y1 >= jersey_y2 or jersey_x1 >= jersey_x2:
        return ""
    jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
    gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = reader.readtext(binary, detail=0, text_threshold=0.7)
    if results:
        number = ''.join([c for c in results[0] if c.isdigit()])
        return number
    return ""

def get_team_color(frame, x1, y1, x2, y2):
    """Определяет цвет майки."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (128, 128, 128)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = np.mean(hsv[:, :, 0])
    if 200 < hue < 250:
        return (255, 0, 0)   # Синяя
    elif 20 < hue < 30:
        return (0, 255, 255) # Жёлтая
    else:
        return (128, 128, 128) # Нейтральный

def calculate_metrics(player_id, center):
    """Рассчитывает скорость и дистанцию."""
    x, y = center
    if player_id not in player_positions:
        player_positions[player_id] = [center]
        player_distances[player_id] = 0.0
        player_speeds[player_id] = 0.0
        return 0.0, 0.0
    last_x, last_y = player_positions[player_id][-1]
    dx = (x - last_x) * PIXEL_TO_METER
    dy = (y - last_y) * PIXEL_TO_METER
    distance_m = (dx**2 + dy**2)**0.5
    speed_m_s = distance_m * FPS
    player_distances[player_id] += distance_m
    player_speeds[player_id] = speed_m_s
    player_positions[player_id].append(center)
    return speed_m_s, player_distances[player_id]

def create_heatmap():
    """Создаёт тепловую карту."""
    heatmap = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
    for player_id, positions_list in player_positions.items():
        for (x, y) in positions_list:
            if 0 <= int(y) < FRAME_HEIGHT and 0 <= int(x) < FRAME_WIDTH:
                heatmap[int(y), int(x)] += 1.0
    heatmap_vis = cv2.resize(heatmap, (FRAME_WIDTH, FRAME_HEIGHT))
    if heatmap_vis.max() > 0:
        heatmap_vis = np.uint8(255 * heatmap_vis / heatmap_vis.max())
    else:
        heatmap_vis = np.uint8(heatmap_vis)
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_path = r"C:/opencv/env/data/heatmap.jpg"
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    cv2.imwrite(heatmap_path, heatmap_vis)
    print(f"The heat map is saved: {heatmap_path}")

def extract_frames(video_path, output_folder, save_every=10):
    """Извлекает кадры из видео."""
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return False
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    os.makedirs(output_folder, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % save_every == 0:
            output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Frames are extracted: {saved_count}")
    return True

def track_players():
    """Трекинг игроков, распознавание номеров, детекция шайбы."""
    os.makedirs(TRACKS_FOLDER, exist_ok=True)
    os.makedirs(TRACKED_FOLDER, exist_ok=True)
    frame_files = sorted([f for f in os.listdir(FRAMES_FOLDER) if f.lower().endswith('.jpg')])
    for frame_file in frame_files:
        frame_path = os.path.join(FRAMES_FOLDER, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        results_players = model_players(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(results_players[0])
        detections = detections[detections.class_id == 0]  # Только люди
        tracks = tracker.update_with_detections(detections)
        frame_copy = frame.copy()
        frame_num = int(os.path.splitext(frame_file)[0].split('_')[1])
        frame_data = {
            "frame_id": frame_num,
            "timestamp_sec": frame_num / 30.0,
            "players": [],
            "puck": []
        }

        # --- Обработка игроков ---
        for i in range(len(tracks)):
            x1, y1, x2, y2 = tracks.xyxy[i].astype(int)
            track_id = int(tracks.tracker_id[i])
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            speed, distance = calculate_metrics(track_id, center)
            color = get_team_color(frame, x1, y1, x2, y2)
            number = extract_jersey_number(frame, x1, y1, x2, y2)
            label_parts = [f"ID:{track_id}"]
            if number:
                label_parts.append(f"#{number}")
            if speed:
                label_parts.append(f"{speed:.1f}m/s")
            label = " ".join(label_parts)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            frame_data["players"].append({
                "player_id": track_id,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(center[0]), float(center[1])],
                "speed_m_s": speed,
                "distance_m": distance,
                "number": number,
                "team_color": "blue" if color == (255, 0, 0) else "yellow"
            })

        # --- Детекция шайбы ---
        results_puck = model_puck(frame, verbose=False)
        for result in results_puck:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Puck {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    frame_data["puck"].append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf
                    })

        # --- Сохранение кадра и JSON ---
        output_path = os.path.join(TRACKED_FOLDER, frame_file)
        cv2.imwrite(output_path, frame_copy)
        json_path = os.path.join(TRACKS_FOLDER, f"frame_{frame_num:04d}.json")
        with open(json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)

        # --- Отладочное изображение ---
        if frame_data["puck"]:
            debug_path = os.path.join(TRACKED_FOLDER, "debug_puck.jpg")
            cv2.imwrite(debug_path, frame_copy)
            print(f"Debugging image with a washer: {debug_path}")

        print(f"Processed: {frame_file} | players: {len(tracks)} | pucks: {len(frame_data['puck'])}")

    # --- Сохранение статистики ---
    stats_df = pd.DataFrame([
        {"Player ID": pid, "Distance (m)": round(dist, 2), "Max Speed (m/s)": round(player_speeds[pid], 2)}
        for pid, dist in player_distances.items()
    ])
    stats_df.to_excel(EXCEL_PATH, index=False)
    print(f"Statistics are saved: {EXCEL_PATH}")

def visualize_analysis():
    """Создаёт видео с визуализацией."""
    json_files = sorted([f for f in os.listdir(TRACKS_FOLDER) if f.endswith('.json')])
    if not json_files:
        return False
    first_frame_file = json_files[0]
    frame_num = int(first_frame_file.split('_')[1].split('.')[0])
    frame_path = os.path.join(TRACKED_FOLDER, f"frame_{frame_num:04d}.jpg")
    first_frame = cv2.imread(frame_path)
    if first_frame is None:
        return False
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))
    for jf in json_files:
        data = json.load(open(os.path.join(TRACKS_FOLDER, jf)))
        frame = cv2.imread(os.path.join(TRACKED_FOLDER, f"frame_{data['frame_id']:04d}.jpg"))
        if frame is None:
            continue
        cv2.rectangle(frame, (w - 300, h - 120), (w - 20, h - 20), (0, 0, 0), -1)
        cv2.putText(frame, f"Time: {data['timestamp_sec']:.1f}s", (w - 290, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Players: {len(data['players'])}", (w - 290, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.write(frame)
    out.release()
    print(f"Video saved: {OUTPUT_VIDEO}")

def save_final_stats():
    """Сохраняет JSON и Excel."""
    stats = {
        "total_frames_processed": len([f for f in os.listdir(TRACKS_FOLDER) if f.endswith('.json')]),
        "players_tracked": list(player_distances.keys()),
        "distance_per_player": {str(pid): round(dist, 2) for pid, dist in player_distances.items()},
        "max_speed_per_player": {str(pid): round(player_speeds[pid], 2) for pid in player_speeds}
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {STATS_JSON}")

def team_analysis():
    """Анализ по командам."""
    team_distance = {"blue": 0.0, "yellow": 0.0}
    team_max_speed = {"blue": 0.0, "yellow": 0.0}
    for player_id, dist in player_distances.items():
        team_distance["blue"] += dist
        if player_speeds[player_id] > team_max_speed["blue"]:
            team_max_speed["blue"] = player_speeds[player_id]
    df = pd.DataFrame([{
        "Team": "Blue", "Total Distance (m)": round(team_distance["blue"], 2),
        "Max Speed (m/s)": round(team_max_speed["blue"], 2)
    }])
    df.to_excel(TEAM_STATS_PATH, index=False)
    print(f"Team statistics: {TEAM_STATS_PATH}")

def generate_report():
    """Генерирует HTML-отчёт."""
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    try:
        df = pd.read_excel(TEAM_STATS_PATH)
        plt.figure(figsize=(8, 5))
        plt.bar(df['Team'], df['Total Distance (m)'], color=['blue'], alpha=0.7)
        plt.title('Total Distance by Team')
        plt.ylabel('Distance (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_FOLDER, "distance_chart.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating chart: {e}")
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Hockey Match Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #1a3e6a; }}
            img {{ max-width: 100%; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Hockey Match Analysis Report</h1>
        <h2>Total Distance by Team</h2>
        <img src="report_images/distance_chart.png" alt="Distance">
    </body>
    </html>
    """
    with open(HTML_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report: {HTML_REPORT_PATH}")

def main():
    """Полный пайплайн."""
    print("Launching...")
    extract_frames(VIDEO_PATH, FRAMES_FOLDER)
    track_players()
    visualize_analysis()
    save_final_stats()
    create_heatmap()
    team_analysis()
    generate_report()
    print("Ready!")

if __name__ == "__main__":
    main()
