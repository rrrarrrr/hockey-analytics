# visualize_analysis.py
"""
Визуализирует трекинг с метриками: скорость, дистанция, владение.
Создаёт видео с панелями слева и справа.
"""

import cv2
import os
import json
import numpy as np

# --- Пути ---
TRACKS_FOLDER = r"C:/opencv/env/data/tracks"
FRAMES_FOLDER = r"C:/opencv/env/data/frames"
OUTPUT_VIDEO = r"C:/opencv/env/data/visualizations/hockey_analysis.mp4"
STATS_JSON = r"C:/opencv/env/data/visualizations/stats.json"

# --- Параметры ---
FPS = 30
FRAME_WIDTH = 784   # Из лога: видео 784x442
FRAME_HEIGHT = 440  # Округляем 442 до 440 (должно быть кратно 2 для видео)

# --- Цвета ---
COLORS = {
    "team_A": (255, 0, 0),      # Синяя команда
    "team_B": (0, 255, 255),    # Жёлтая команда
    "text": (255, 255, 255),
    "bg": (0, 0, 0)
}

# --- История для анализа ---
positions = {}
distances = {}
speeds = {}


def load_all_tracks():
    """Загружает все .json файлы из папки tracks."""
    tracks = []
    if not os.path.exists(TRACKS_FOLDER):
        print(f" Папка не найдена: {TRACKS_FOLDER}")
        return tracks

    json_files = sorted([f for f in os.listdir(TRACKS_FOLDER) if f.endswith(".json")])
    for jf in json_files:
        file_path = os.path.join(TRACKS_FOLDER, jf)
        if os.path.getsize(file_path) == 0:
            print(f" Пропущен пустой файл: {file_path}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tracks.append(data)
        except json.JSONDecodeError as e:
            print(f"Ошибка в JSON {file_path}: {e}")
            continue
    return tracks


def calculate_speed_and_distance(player_id, center, fps=30):
    """Рассчитывает скорость (м/с) и общую дистанцию."""
    pixel_to_meter = 0.05  # 1 пиксель = 5 см

    if player_id not in positions:
        positions[player_id] = [center]
        distances[player_id] = 0.0
        speeds[player_id] = 0.0
        return 0.0, 0.0

    last_pos = positions[player_id][-1]
    dx = (center[0] - last_pos[0]) * pixel_to_meter
    dy = (center[1] - last_pos[1]) * pixel_to_meter
    distance_m = (dx**2 + dy**2)**0.5
    speed_m_s = distance_m * fps

    distances[player_id] += distance_m
    speeds[player_id] = speed_m_s
    positions[player_id].append(center)

    return speed_m_s, distances[player_id]


def extract_jersey_number(frame, x1, y1, x2, y2):
    """
    Улучшенное распознавание номера на майке.
    """
    h = y2 - y1
    w = x2 - x1

    # Увеличим область чуть ниже (номера обычно на спине)
    jersey_y1 = int(y1 + 0.6 * h)
    jersey_y2 = int(y2 - 0.1 * h)  # не до самого низа
    jersey_x1 = int(x1 + 0.15 * w)
    jersey_x2 = int(x2 - 0.15 * w)

    # Проверка границ
    jersey_y1 = max(jersey_y1, 0)
    jersey_y2 = min(jersey_y2, frame.shape[0])
    jersey_x1 = max(jersey_x1, 0)
    jersey_x2 = min(jersey_x2, frame.shape[1])
    if jersey_y1 >= jersey_y2 or jersey_x1 >= jersey_x2:
        return ""

    # Кроп
    jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

    # Увеличим размер (для лучшего OCR)
    scale = 2
    jersey_crop = cv2.resize(jersey_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # В BGR → GRAY
    gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)

    # Улучшение контраста: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    gray = clahe.apply(gray)

    # Бинаризация
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Удаление шума
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Распознавание
    results = reader.readtext(
        binary,
        detail=0,
        paragraph=False,
        text_threshold=0.7,
        low_text=0.4,
        contrast_ths=0.1,
        allowlist='0123456789'  # только цифры
    )

    if results:
        number = ''.join([c for c in results[0] if c.isdigit()])
        if len(number) <= 2:  # хоккей: 1–99
            return number
    return ""
def get_team_color(frame, x1, y1, x2, y2):
    """
    Точнее определяет команду по цвету майки.
    Использует HSV + статистику доминирующего цвета.
    """
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (128, 128, 128)

    # Уменьшим шум — размытие
    crop = cv2.GaussianBlur(crop, (5,5), 0)

    # В HSV
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Разделим на области: верх (плечи), низ (номер)
    h, w = hsv.shape[:2]
    upper = hsv[:h//2, :]
    lower = hsv[h//2:, :]

    # Подсчёт пикселей в диапазонах
    blue_mask = cv2.inRange(upper, (100, 50, 50), (140, 255, 255))   # синий
    yellow_mask = cv2.inRange(upper, (20, 100, 100), (35, 255, 255))  # жёлтый

    blue_pixels = cv2.countNonZero(blue_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    total = upper.size // 3  # кол-во пикселей

    if blue_pixels > yellow_pixels and blue_pixels / total > 0.1:
        return (255, 0, 0)   # Синяя
    elif yellow_pixels > blue_pixels and yellow_pixels / total > 0.1:
        return (0, 255, 255) # Жёлтая
    else:
        return (128, 128, 128) # Не определено


def main():
    tracks = load_all_tracks()
    if not tracks:
        print(" Нет данных треков. Запусти track_players.py")
        return

    # Создаём папку для визуализаций
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Создаём VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    if not out.isOpened():
        print(" Ошибка: Не удалось создать видео.")
        return

    for frame_data in tracks:
        frame_num = frame_data["frame_id"]
        frame_path = os.path.join(FRAMES_FOLDER, f"frame_{frame_num:04d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f" Не удалось загрузить: {frame_path}")
            continue

        # Изменяем размер, если нужно
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Рисуем каждого игрока
        for player in frame_data["players"]:
            pid = player["player_id"]
            x1, y1, x2, y2 = map(int, player["bbox"])
            center = player["center"]

            color = get_team_color(pid)
            speed, distance = calculate_speed_and_distance(pid, center)

            # Рамка
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Подпись
            label = f"ID:{pid} {speed:.1f}m/s"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Правая панель (внизу справа)
        cv2.rectangle(frame, (FRAME_WIDTH - 300, FRAME_HEIGHT - 120), (FRAME_WIDTH - 20, FRAME_HEIGHT - 20), COLORS["bg"], -1)
        cv2.putText(frame, f"Time: {frame_data['timestamp_sec']:.1f}s", (FRAME_WIDTH - 290, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)
        cv2.putText(frame, f"Players: {len(frame_data['players'])}", (FRAME_WIDTH - 290, FRAME_HEIGHT - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)
        cv2.putText(frame, f"FPS: {FPS}", (FRAME_WIDTH - 290, FRAME_HEIGHT - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)

        # Левая панель (внизу слева)
        cv2.rectangle(frame, (20, FRAME_HEIGHT - 150), (300, FRAME_HEIGHT - 20), COLORS["bg"], -1)
        cv2.putText(frame, f"Total Players: {len(frame_data['players'])}", (30, FRAME_HEIGHT - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)

        # Средняя скорость
        current_speeds = [speeds.get(p["player_id"], 0) for p in frame_data["players"]]
        if current_speeds:
            avg_speed = sum(current_speeds) / len(current_speeds)
            cv2.putText(frame, f"Avg Speed: {avg_speed:.1f} m/s", (30, FRAME_HEIGHT - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text"], 1)

        # Записываем кадр
        out.write(frame)

    # Завершаем запись
    out.release()

    # Сохраняем итоговую статистику
    os.makedirs(os.path.dirname(STATS_JSON), exist_ok=True)
    stats = {
        "total_frames_processed": len(tracks),
        "players_tracked": list(distances.keys()),
        "distance_per_player": {str(pid): round(dist, 2) for pid, dist in distances.items()},
        "max_speed_per_player": {str(pid): round(speeds[pid], 2) for pid in speeds}
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Видео с анализом сохранено: {OUTPUT_VIDEO}")
    print(f"Статистика сохранена: {STATS_JSON}")


if __name__ == "__main__":
    main()
