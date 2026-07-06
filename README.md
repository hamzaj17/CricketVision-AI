# 🏏 CricketVision AI

An AI-powered cricket analytics system built using **YOLOv8** that detects and tracks cricket objects including **players, bats, balls, and stumps** from images and videos.

The project was developed to explore real-time sports computer vision and serves as a foundation for advanced cricket analytics such as player tracking, ball trajectory estimation, and automated match analysis.

---

# 📌 Features

- 🎯 Real-time object detection
- 🏏 Detects:
  - Ball
  - Bat
  - Player
  - Stumps
- 📹 Video inference
- 🖼️ Image inference
- 📊 Custom trained YOLOv8 model
- ⚡ GPU accelerated inference (CUDA supported)
- 📈 Dataset preparation and training pipeline

---

# 🖼️ Sample Results

*(Add screenshots or GIFs here)*

Example:

```
assets/demo.gif
assets/result1.png
assets/result2.png
```

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Object Detection | YOLOv8 (Ultralytics) |
| Image Processing | OpenCV |
| Visualization | Matplotlib |
| Dataset Annotation | Roboflow |
| Dataset Management | Custom Python Scripts |

---

# 📂 Project Structure

```
CricketVision_AI
│
├── datasets/
│   ├── cricket_dataset_final/
│   └── ...
│
├── models/
│   ├── cricketvision_final.pt
│   └── cricketvision_v1.pt
│
├── scripts/
│   ├── train_final.py
│   ├── test_video.py
│   ├── track_video.py
│   ├── predict.py
│   ├── extract_frames.py
│   └── split_dataset.py
│
├── test_videos/
├── video_data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The dataset consists of manually annotated cricket images collected from multiple broadcast videos.

### Classes

- Ball
- Bat
- Player
- Stumps

### Dataset Statistics

| Split | Images |
|--------|---------|
| Train | 498 |
| Validation | 62 |
| Test | 63 |
| **Total** | **623** |

Annotations were created using **Roboflow** and additional custom annotation tools.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CricketVision_AI.git
```

Move into the project

```bash
cd CricketVision_AI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Training

Train the final model

```bash
python scripts/train_final.py
```

The trained weights will be saved inside

```
runs/detect/
```

---

# 🎥 Test on a Video

Place your video inside

```
test_videos/
```

Run

```bash
python scripts/test_video.py
```

Output videos will be saved automatically inside

```
runs/detect/
```

---

# 📷 Test on Images

```bash
python scripts/predict.py
```

---

# 📈 Current Performance

Validation Results

| Metric | Score |
|---------|--------|
| Precision | 84.1% |
| Recall | 77.7% |
| mAP@50 | **82.1%** |
| mAP@50-95 | **49.9%** |

### Per-Class Performance

| Class | mAP@50 |
|---------|---------|
| Ball | 80.9% |
| Bat | 77.9% |
| Player | 94.5% |
| Stumps | 75.0% |

---

# ⚠ Current Limitations

The model performs well on broadcast cricket footage but still faces challenges in some scenarios:

- Small and fast-moving balls
- Occluded stumps
- Motion blur
- Camera zoom transitions
- Replay scenes
- Player ID consistency during tracking

These limitations can be addressed with a larger and more diverse training dataset.

---

# 🔮 Future Improvements

- DeepSORT / ByteTrack integration
- Stable player identity tracking
- Ball trajectory estimation
- Pitch mapping
- Shot classification
- Stroke analysis
- Wagon wheel generation
- Automatic highlights
- Scoreboard OCR
- LBW assistance
- Hawkeye-style visualization

---

# 👨‍💻 Author

**Hamza Bin Javed**

Bachelor of Artificial Intelligence

GitHub:
https://github.com/hamzaj17

---

# ⭐ Acknowledgements

- Ultralytics YOLOv8
- Roboflow
- PyTorch
- OpenCV

---

## 📄 License

This project is intended for educational and research purposes.