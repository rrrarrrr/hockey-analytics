# 🏒 Hockey Player & Puck Tracking System

A complete computer vision pipeline for **automated hockey match analysis** using modern AI tools.  
Detects players, tracks movement, recognizes jersey numbers, detects the puck, and generates detailed reports.

![Hockey Analytics](https://via.placeholder.com/800x400?text=Hockey+Tracking+Demo)  
*Example: Players tracked with IDs, speed, and team colors*

## 🚀 Features

- ✅ **Player Detection & Tracking** — YOLOv8 + ByteTrack
- 🔢 **Jersey Number Recognition** — using `easyocr`
- 🎨 **Team Color Detection** — blue vs yellow jerseys
- 🏐 **Puck Detection** — custom-trained model (YOLO)
- 📊 **Player Metrics** — speed (m/s), distance (m), heatmaps
- 📈 **Excel & JSON Reports** — per-player and team stats
- 🌐 **HTML Dashboard** — interactive report with charts
- 🎥 **Annotated Video Output** — real-time stats overlay

## 📁 Project Structure
```bash
hockey-analytics/
├── main.py # Full pipeline: extract → track → visualize
├── track_players.py # Advanced tracking with OCR and team detection
├── train_puck.py # Train YOLO model on puck dataset
├── retrain_puck.py # Retrain existing model with new data
├── data.yaml # Dataset config for YOLO training
├── requirements.txt # Python dependencies
├── README.md # This file
└── LICENSE # MIT License
```

## 🛠️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/rrrarrrr/hockey-analytics.git
cd hockey-analytics
```

###2. Create virtual environment and install dependencies
```bash
python -m venv env
env\Scripts\activate    # Windows
# source env/bin/activate   # Linux/Mac

pip install -r requirements.txt
```
###3. Add your video
Place your hockey video in:
```bash
data/raw/hockey_match.mp4
```
###4. Run the full pipeline
```bash
python main.py
```
###📈 Output
After running, you'll get:
```bash
data/tracked/ — frames with bounding boxes and labels
data/tracks/ — JSON files with player and puck positions
data/hockey_analysis.mp4 — annotated video with live stats
data/stats.xlsx — Excel report with speed and distance
data/heatmap.jpg — player activity heatmap
data/hockey_report.html — interactive HTML dashboard
```

###🏁 Train the Puck Detector
To improve puck detection:
```bash
python train_puck.py
```
Make sure you have:
```bash
data/puck_train/images/ — images with pucks
data/puck_train/labels/ — YOLO-format annotations (from LabelImg)
```
Use the trained model in main.py:
```bash
model_puck = YOLO('runs/detect/puck_detection/weights/best.pt')
```
###📄 License
MIT License — free for personal and commercial use.

