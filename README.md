# RoadVision — Smart Road Rule Driver Assistance System with Voice Alerts

## Introduction

**RoadVision** is a smart driver-assistance system that detects traffic signs in real time and alerts the driver with a **voice message**, helping keep the driver's attention and awareness of road rules while driving.

The system is tailored specifically for **Jordanian streets**, since the underlying object detection model was trained on a **custom dataset built from Google Street View imagery of Jordan**, manually annotated for object detection. The model was tested live inside a car on real Jordanian roads to evaluate its real-world performance.

---

## Dataset

- **Size:** 10,000+ images
- **Classes:** 25 different traffic sign types
- **Source:** Collected from Google Street View (Jordan)
- **Annotation:** Manually labeled for object detection tasks

---

## Model

- **Architecture:** YOLO26s (latest YOLO version at the time of development)
- **Why YOLO26s:** Optimized for edge deployment with fast inference, making it suitable for real-time use inside a moving vehicle.

---

## Deployment

The trained model was exported to **ONNX format** and integrated into a Python application that:

- Runs real-time inference on the video feed
- Triggers a **voice alert in Arabic** to inform the driver of the detected traffic rule

Special handling was implemented for cases where multiple traffic signs appear **in quick succession or simultaneously**:

- A **priority queue** to determine which alert should be announced first
- **Timeout** and **cooldown** logic to prevent overlapping or repetitive alerts

---

## Experiments & Results

The system was deployed in a car and tested across **3 real-world experiments** in different locations:

| Experiment Location | Duration | Traffic Sign Types | Appeared | Captured (TP) | False Alerts (FP) |
|----------------------|----------|---------------------|----------|----------------|--------------------|
| Al-Tafila            | 20 mins  | 11                  | 46       | 45             | 5                  |
| Amman (Route 1)      | 30 mins  | 16                  | 83       | 83             | 9                  |
| Amman (Route 2)      | 33 mins  | 18                  | 113      | 109            | 10                 |
| **Total**            | **83 mins** | 21                | **242**  | **237**        | **24**             |

**Overall performance:**
- Recall ≈ 97.9% (237 / 242 signs correctly detected)
- Precision ≈ 90.8% (237 / 261 total alerts triggered)

---

## Repository Contents

```
RoadVision/
├── training_notebooks/
│   ├── session_1.ipynb
│   └── session_2.ipynb
├── deployment/
│   ├── traffic_sign_voice.py     # Main deployment script
│   ├── best.onnx                 # Trained model (ONNX format)
│   └── voice_messages/           # Arabic voice alert audio files
├── videos/
│   ├── testing_vids/             # Model bounding box evaluation
│   └── experiments/               # Real-time in-car experiment footage
└── documentation.pdf             # Full project documentation
```

### Notes
- **Training Notebooks (2 sessions):** Contain the full model training pipeline across two sessions.
- **Deployment Code:** `traffic_sign_voice.py` runs the ONNX model and handles voice alert logic (priority queue, timeout, cooldown). It is located in the same folder as `best.onnx` and the voice message files.
- **Camera Input:** During experiments, the **iVCam** app was used to connect the laptop to the camera.
- **Videos:**
  - *Testing video* — demonstrates model bounding box detection performance.
  - *Experiments video* — recorded during the real-time in-car experiments.
- **Documentation:** A detailed PDF report covering the full project is included.
