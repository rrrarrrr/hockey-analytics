# semi_automatic_puck_tracker.py
import cv2
import numpy as np
import os

# --- Пути ---
FRAME_FOLDER = r"C:/opencv/env/data/frames"
OUTPUT_FOLDER = r"C:/opencv/env/data/puck_tracked"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Параметры ---
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
PADDING = 20  # Дополнительная область вокруг шайбы

# --- Переменные ---
puck_template = None  # Шаблон шайбы
puck_position = None  # Текущее положение шайбы
puck_roi = None       # ROI для поиска
traj = []             # Траектория

def select_puck(event, x, y, flags, param):
    global puck_template, puck_position, puck_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Шайба выбрана: x={x}, y={y}")
        frame = param
        h, w = frame.shape[:2]
        x1 = max(0, x - PADDING)
        x2 = min(w, x + PADDING)
        y1 = max(0, y - PADDING)
        y2 = min(h, y + PADDING)
        puck_template = frame[y1:y2, x1:x2].copy()
        puck_position = (x, y)
        # Начальный ROI для поиска
        puck_roi = (x1, y1, x2 - x1, y2 - y1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Select Puck", frame)

def track_with_optical_flow(prev_gray, curr_gray, point):
    """Отслеживает точку с помощью optical flow."""
    point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
    next_point, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, point, None, **LK_PARAMS)
    if status[0] == 1:
        return tuple(next_point[0][0])
    return None
import cv2

# Создаём фильтр Калмана
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10

def main():
    global puck_position

    frame_files = sorted([f for f in os.listdir(FRAME_FOLDER) if f.endswith(".jpg")])
    if not frame_files:
        print("There are no frames")
        return

    # --- Этап 1: Выбор шайбы на первом кадре ---
    first_frame_path = os.path.join(FRAME_FOLDER, frame_files[91])
    frame = cv2.imread(first_frame_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Select Puck")
    cv2.setMouseCallback("Select Puck", select_puck, frame)
    cv2.imshow("Select Puck", frame)
    print("Left-click on the puck. Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if puck_template is None:
        print("The puck is not selected")
        return

    # В цикле:
    measured = np.array([x, y], np.float32)
    kf.correct(measured)
    predicted = kf.predict()
    smooth_x, smooth_y = predicted[0], predicted[1]
    # --- Этап 2: Трекинг по остальным кадрам ---
    prev_gray = gray
    traj.append(puck_position)

    res = cv2.matchTemplate(...)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.5:  # Порог качества
        # Шаблон не найден — пропустить или использовать предсказание
        new_pos = predicted[:2] if 'predicted' in locals() else puck_position
    else:
        new_pos = (search_x1 + max_loc[0], search_y1 + max_loc[1])

    for frame_file in frame_files[1:]:
        frame_path = os.path.join(FRAME_FOLDER, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Попробуем optical flow
        new_pos = track_with_optical_flow(prev_gray, curr_gray, puck_position)
        if new_pos is None:
            print(f"Optical flow failed for {frame_file}")
            new_pos = puck_position  # Используем старое положение

        # Уточним с помощью template matching в окрестности
        x, y = int(new_pos[0]), int(new_pos[1])
        search_x1 = max(0, x - 50)
        search_y1 = max(0, y - 50)
        search_x2 = min(frame.shape[1], x + 50)
        search_y2 = min(frame.shape[0], y + 50)
        search_region = curr_gray[search_y1:search_y2, search_x1:search_x2]
        res = cv2.matchTemplate(search_region, cv2.cvtColor(puck_template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        _, _, _, top_left = cv2.minMaxLoc(res)
        corrected_x = search_x1 + top_left[0] + PADDING
        corrected_y = search_y1 + top_left[1] + PADDING
        puck_position = (corrected_x, corrected_y)
        traj.append(puck_position)

        # Рисуем результат
        cv2.circle(frame, (corrected_x, corrected_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Puck", (corrected_x + 10, corrected_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        output_path = os.path.join(OUTPUT_FOLDER, frame_file)
        cv2.imwrite(output_path, frame)
        print(f"done: {frame_file}")

        prev_gray = curr_gray

    # --- Сохраняем траекторию ---
    import json
    with open("data/puck_trajectory.json", "w") as f:
        json.dump({"trajectory": traj}, f)
    print(" The trajectory of the puck is preserved: data/puck_trajectory.json")

if __name__ == "__main__":
    main()
