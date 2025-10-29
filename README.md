# 🧠 Real-Time Violence Detection using Pose Estimation

This project detects **violent actions in real-time video streams** using **human pose estimation** and **deep learning-based temporal analysis**.  
It leverages **YOLOv8 Pose** for keypoint extraction and a **custom trained MobileNet + LSTM based model** to predict violence from human motion sequences.

---

## 🎬 Demo Video
🎥 [Watch the Demo on YouTube](https://youtu.be/bcP7qmx-6s0)

---

## 🖼️ Sample Outputs

| Normal Activity | Violence Detected |
|-----------------|------------------|
| ![Normal Pose](assets/normal_pose.jpg) | ![Violence Detected](assets/violence_pose.jpg) |

---

## 🧩 Brief Overview

- **Goal:** To detect real-time violent activity in CCTV or live webcam feeds.  
- **Method:**  
  - Extract pose keypoints from each frame using YOLOv8 Pose model.  
  - Collect a short frame sequence (e.g., 10 frames).  
  - Pass the sequence through an **MobileNet + LSTM model** to analyze temporal motion patterns.  
  - Display “Violence” or “No Violence” on the video feed based on prediction.  

---

## ⚙️ Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Deep Learning | TensorFlow / Keras |
| Pose Estimation | Ultralytics YOLOv8-Pose |
| Model Serving | MobileNetV2 + LSTM |
| Data Handling | NumPy, Pandas |

---

## 🧠 Models Used

| Model | Purpose |
|--------|----------|
| **YOLOv8n-pose.pt** | Extracts 17 human body keypoints per frame. |
| **MobileNet + LSTM Sequential Model** | Classifies sequence of pose coordinates into “Violence” or “No Violence.” |

---

## 🧺 Dataset

- **Source:** [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset).
- **Structure:**  
  - Videos divided into *Violence* and *Non-Violence* classes.  
  - Each video is processed to extract pose keypoints for temporal modeling.  

---

## 🚧 Challenges Faced

1. **Memory Handling:** Extracting and storing pose keypoints from hundreds of high-resolution frames caused high memory usage.  
2. **Data Transformation:** Large `.npy` arrays for keypoints sequences were difficult to manage during training.  
3. **Frame Synchronization:** Maintaining temporal consistency between sequences for accurate violence detection.  
4. **Model Optimization:** Needed to balance accuracy and inference speed for real-time detection.

---

## 🧩 How It Works

1. **Frame Capture:** Captures live feed from webcam using OpenCV.  
2. **Pose Extraction:** YOLOv8 Pose detects keypoints for each person.  
3. **Sequence Formation:** Collects last 10 frames’ keypoints.  
4. **Prediction:** The LSTM model predicts if the sequence shows violent motion.  
5. **Display:** Overlays the prediction label and confidence on the live video.  

---

## ▶️ Run the Project

```bash
# Clone the repo
git clone https://github.com/yourusername/violence-detection-pose.git
cd violence-detection-pose

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## 📚 Folder Structure

```
📁 src/
 ┣ 📂 components/
 ┃ ┗ data_transformation.py
 ┣ 📂 pipeline/
 ┃ ┗ predict_pipeline.py
 ┣ 📂 models/
 ┃ ┗ violence_lstm_model.h5
 ┣ app.py
 ┣ README.md
 ┣ requirements.txt
```

---

## 👨‍💻 Author

**Om — Data Scientist**  
📧 omtambat284@gmail.com.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)

---

*Last updated: 2025-10-28*
