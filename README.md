# 🏏 CricketVision AI

CricketVision AI is a computer vision project built using **YOLOv8** for detecting key cricket objects—including **players, bats, balls, and stumps**—from cricket broadcast images and videos.

The project includes a complete end-to-end pipeline for **frame extraction, dataset creation, manual annotation, dataset preparation, model training, and inference**. It serves as a foundation for future cricket analytics such as object tracking, ball trajectory estimation, player movement analysis, and automated match insights.

---

# 🎥 Demo

### Object Detection on Unseen Cricket Footage

![Demo](assets/demo.gif)

---

# ✨ Features

- 🎯 Custom YOLOv8 object detector
- 🏏 Detects:
  - Ball
  - Bat
  - Player
  - Stumps
- 📹 Video inference
- 🖼️ Image inference
- 📊 Complete model training pipeline
- 🛠️ Custom dataset preparation scripts
- ⚡ GPU accelerated training and inference
- 📦 Automated dataset splitting
- 📝 Custom annotation tool for label correction

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Visualization | Matplotlib |
| Dataset Annotation | Roboflow + Custom Annotation Tool |
| Dataset Preparation | Custom Python Scripts |

---

# 📂 Project Structure

```text
CricketVision_AI
│
├── assets/
│   ├── demo.gif
│   └── confusion_matrix.png
│
├── models/
│   └── cricketvision_final.pt
│   
├── scripts/
│   ├── extract_frames.py
│   ├── annotation_tool.py
│   ├── split_dataset.py
│   ├── train_final.py
│   ├── predict.py
│   ├── test_video.py
│   └── track_video.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

One of the primary goals of this project was to build a **custom cricket object detection dataset** rather than relying entirely on existing public datasets.

### Dataset Creation Process

1. Cricket broadcast videos were collected.
2. Frames were extracted using custom Python scripts.
3. An initial subset of images was manually annotated in Roboflow.
4. A preliminary YOLOv8 model was trained on this subset.
5. The trained model was then used to generate initial predictions on the remaining images.
6. A custom annotation tool was developed to manually review, correct, and improve these predictions.
7. All corrected annotations were merged into a final dataset and randomly split into training, validation, and testing sets.

This semi-automated workflow significantly reduced annotation time while maintaining annotation quality.

### Classes

- 🏏 Ball
- 🏏 Bat
- 🧍 Player
- 🎯 Stumps

### Final Dataset Statistics

| Split | Images |
|--------|---------|
| Train | 498 |
| Validation | 62 |
| Test | 63 |
| **Total** | **623** |

All annotations were manually verified before training the final model.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/hamzaj17/CricketVision_AI.git
```

Move into the project

```bash
cd CricketVision_AI
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

# 🧠 Training

Train the final YOLOv8 model

```bash
python scripts/train_final.py
```

The trained model will be saved inside

```text
runs/detect/CricketVision_Final/
```

---

# 🎥 Video Inference

Place the test video inside

```text
test_videos/
```

Run

```bash
python scripts/test_video.py
```

The output video will be saved automatically.

---

# 🖼️ Image Inference

Run

```bash
python scripts/predict.py
```

---

# 📈 Results

The final detector was trained on **623 manually verified cricket images**.

### Overall Validation Performance

| Metric | Score |
|---------|--------|
| Precision | **84.1%** |
| Recall | **77.7%** |
| mAP@50 | **82.1%** |
| mAP@50-95 | **49.9%** |

### Per-Class Performance

| Class | Precision | Recall | mAP@50 |
|---------|----------:|--------:|--------:|
| Ball | 100.0% | 70.3% | 80.9% |
| Bat | 76.6% | 75.9% | 77.9% |
| Player | 92.3% | 93.0% | 94.5% |
| Stumps | 67.3% | 71.7% | 75.0% |

The model performs strongly on broadcast cricket footage, with excellent player detection and good overall performance for the remaining classes. Small objects such as the ball and stumps remain the most challenging due to motion blur, occlusion, and their limited size within the frame.

---

# ⚠ Current Limitations

The current version performs well on standard broadcast footage but still has some limitations:

- Small and fast-moving cricket balls
- Partial stump occlusion
- Motion blur
- Sudden camera zooms and transitions
- Replay sequences
- Stable object tracking has not yet been integrated

---

# 🚀 Future Improvements

The following features are planned for future development:

- ByteTrack / DeepSORT integration
- Stable player identity tracking
- Ball trajectory estimation
- Player movement analysis
- Cricket pitch mapping
- Shot direction visualization
- Wagon wheel generation
- Heatmaps
- Automatic highlight generation
- Scoreboard OCR
- Real-time cricket analytics dashboard

---

# 👨‍💻 Author

**Hamza Bin Javed**

GitHub: https://github.com/hamzaj17

---

# 🙏 Acknowledgements

This project was built using several excellent open-source tools and libraries.

- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Roboflow
- Matplotlib

---

# 📄 License

This project is released for educational, research, and portfolio purposes.