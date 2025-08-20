# plot_puck_trajectory.py
import cv2
import json
import os
import numpy as np

# --- Пути ---
PUCK_DATA_PATH = r"C:/opencv/env/data/puck_manual.json"
FRAME_FOLDER = r"C:/opencv/env/data/frames"
OUTPUT_PATH = r"C:/opencv/env/data/puck_trajectory.jpg"

# --- Параметры ---
FRAME_REFERENCE = "frame_0500.jpg"  # Кадр для визуализации
TRAJECTORY_COLOR = (0, 0, 255)      # Красный (BGR)
RADIUS = 3
THICKNESS = -1  # Залитый круг
LINE_COLOR = (255, 0, 0)            # Синяя линия (BGR)
LINE_THICKNESS = 1

def main():
    # 1. Загружаем данные шайбы
    if not os.path.exists(PUCK_DATA_PATH):
        print("Error: puck_manual.json not found. Run manual_puck_input.py first.")
        return

    with open(PUCK_DATA_PATH, 'r') as f:
        try:
            puck_data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Cannot decode puck_manual.json. File is corrupted.")
            return

    if not puck_data:
        print("Error: No puck data found in puck_manual.json.")
        return

    print(f"Loaded {len(puck_data)} puck positions.")

    # 2. Загружаем опорный кадр
    ref_frame_path = os.path.join(FRAME_FOLDER, FRAME_REFERENCE)
    frame = cv2.imread(ref_frame_path)

    if frame is None:
        print(f"Error: Cannot load reference frame: {ref_frame_path}")
        print("Using blank image.")
        # Создаём пустое изображение (под размер льда)
        frame = np.ones((442, 784, 3), dtype=np.uint8) * 240

    # 3. Собираем координаты траектории
    points = []
    for frame_name, pos in puck_data.items():
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            x, y = int(pos[0]), int(pos[1])
            points.append((x, y))

    if len(points) < 1:
        print("Error: No valid puck positions to plot.")
        return

    # 4. Рисуем траекторию
    # Сначала линии
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], LINE_COLOR, LINE_THICKNESS)

    # Потом точки
    for x, y in points:
        cv2.circle(frame, (x, y), RADIUS, TRAJECTORY_COLOR, THICKNESS)

    # 5. Сохраняем
    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, frame)
    print(f"Puck trajectory saved: {OUTPUT_PATH}")

    # 6. Показываем (опционально)
    cv2.imshow("Puck Trajectory", frame)
    print("Press any key on image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
