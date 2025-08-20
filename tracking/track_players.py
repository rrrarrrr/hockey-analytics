# track_players.py
"""
Полный пайплайн: детекция + трекинг игроков.
С распознаванием номеров и шайбы.
"""

import cv2
import os
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv
import easyocr
import pandas as pd

# --- Пути ---
FRAME_FOLDER = r"C:/opencv/env/data/frames"
OUTPUT_FOLDER = r"C:/opencv/env/data/tracked"
TRACKS_FOLDER = r"C:/opencv/env/data/tracks"
EXCEL_PATH = r"C:/opencv/env/data/stats.xlsx"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRACKS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)

# --- Цвета для ID ---
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3), dtype=int)

# --- Глобальные переменные для анализа ---
player_positions = {}
player_distances = {}
player_speeds = {}
PIXEL_TO_METER = 0.05
FPS = 30

# --- Трекер ---
tracker = sv.ByteTrack()

# --- Загрузка модели YOLO ---
print("Downloading YOLOv8...")
model = YOLO('yolov8n.pt')

# --- OCR для номеров ---
print("Loading OCR...")
reader = easyocr.Reader(['en'], gpu=False)

# --- Функции ---

def extract_jersey_number(frame, x1, y1, x2, y2):
    """Распознаёт номер на майке."""
    h = y2 - y1
    w = x2 - x1
    jersey_y1 = int(y1 + 0.6 * h)  # Нижняя треть бокса
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
    print("Creating a heat map...")
    heatmap = np.zeros((442, 784), dtype=np.float32)
    for player_id, positions_list in player_positions.items():
        for (x, y) in positions_list:
            if 0 <= int(y) < 442 and 0 <= int(x) < 784:
                heatmap[int(y), int(x)] += 1.0
    heatmap_vis = cv2.resize(heatmap, (784, 442))
    if heatmap_vis.max() > 0:
        heatmap_vis = np.uint8(255 * heatmap_vis / heatmap_vis.max())
    else:
        heatmap_vis = np.uint8(heatmap_vis)
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_path = r"C:/opencv/env/data/heatmap.jpg"
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    cv2.imwrite(heatmap_path, heatmap_vis)
    print(f"The heat map is saved: {heatmap_path}")

def main():
    print("Starting player tracking...")
    frame_files = sorted([f for f in os.listdir(FRAME_FOLDER) if f.lower().endswith(('.jpg', '.png'))])
    if not frame_files:
        print("There are no frames. Start it up first main.py ")
        return

    for frame_file in frame_files:
        frame_name = os.path.splitext(frame_file)[0]
        frame_num = int(frame_name.split('_')[1])
        frame_path = os.path.join(FRAME_FOLDER, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load: {frame_path}")
            continue

        # Детекция
        results = model(frame, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = detections[detections.class_id == 0]  # Только люди

        # Трекинг
        tracks = tracker.update_with_detections(detections)

        # Подготовка кадра
        frame_copy = frame.copy()
        frame_data = {
            "frame_id": frame_num,
            "timestamp_sec": frame_num / 30.0,
            "players": [],
            "puck": []
        }

        # Обработка треков
        for i in range(len(tracks)):
            x1, y1, x2, y2 = tracks.xyxy[i].astype(int)
            track_id = int(tracks.tracker_id[i])
            confidence = float(tracks.confidence[i])
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
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            frame_data["players"].append({
                "player_id": track_id,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(center[0]), float(center[1])],
                "confidence": confidence,
                "speed_m_s": speed,
                "distance_m": distance,
                "number": number,
                "team_color": "blue" if color == (255, 0, 0) else "yellow" if color == (0, 255, 255) else "unknown"
            })

        # Детекция шайбы
        puck_detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == 32 and conf > 0.5:  # sports ball
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    puck_detections.append([x1, y1, x2, y2])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Puck {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Сохранение шайбы
        for det in puck_detections:
            x1, y1, x2, y2 = det
            frame_data["puck"].append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": 1.0
            })

        # Сохранение
        output_path = os.path.join(OUTPUT_FOLDER, frame_file)
        cv2.imwrite(output_path, frame_copy)

        json_path = os.path.join(TRACKS_FOLDER, f"frame_{frame_num:04d}.json")
        with open(json_path, 'w') as f:
            json.dump(frame_data, f, indent=2)

        print(f" Processed by: {frame_file} | Players: {len(tracks)} | Washers: {len(puck_detections)}")

    # Сохранение статистики
    stats_df = pd.DataFrame([
        {"Player ID": pid, "Distance (m)": round(dist, 2), "Max Speed (m/s)": round(player_speeds[pid], 2)}
        for pid, dist in player_distances.items()
    ])
    stats_df.to_excel(EXCEL_PATH, index=False)
    print(f"Statistics are saved: {EXCEL_PATH}")

    create_heatmap()
    print("Tracking is complete!")

if __name__ == "__main__":
    main()
