# 🎓 Student Activity Detection using YOLO

This repository implements a complete **computer vision pipeline** for detecting, tracking, and analyzing classroom student activities using a YOLO-based deep learning model.

The system:

• merges heterogeneous datasets  
• trains a medium-scale YOLO detector  
• evaluates performance on held-out test data  
• performs video inference with tracking and temporal smoothing  
• logs detections into CSV format  
• conducts post-hoc statistical analytics  

---

---

# 📐 System Architecture

The pipeline consists of six major stages:

1. Dataset Merging & Harmonization  
2. Model Training  
3. YOLO Detection Mathematics  
4. Model Evaluation  
5. Real-Time Inference + Tracking  
6. Post-Inference Analytics  

---

---

# 📂 Repository Structure

├── merge_datasets.py
├── train.py
├── test_model.py
├── inference.py
├── analyze.py
├── datasets/
│ └── merged_dataset/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ ├── val/
│ │ ├── images/
│ │ └── labels/
│ ├── test/
│ │ ├── images/
│ │ └── labels/
│ └── dataset.yaml
├── weights/
│ └── best.pt
├── outputs/
│ ├── output_inference.mp4
│ └── detections_log.csv
└── README.md

---

---

# 1️⃣ Dataset Merging & Preparation

### Script: `merge_datasets.py`

### 🎯 Objective

Combine Dataset-A and Dataset-B into a **single unified dataset** while:

• resolving overlapping class names  
• re-indexing class IDs  
• balancing splits  
• creating Train / Validation / Test folders  
• generating a unified YAML configuration file  

---

## 📊 Split Ratios

| Split | Ratio |
|------|------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## 📂 Output Folder Layout

merged_dataset/
├── train/images
├── train/labels
├── val/images
├── val/labels
├── test/images
├── test/labels
└── dataset.yaml

### `dataset.yaml` Contains:

• relative paths to train/val/test  
• number of classes  
• ordered activity names  

---

---

# 2️⃣ Model Training

### Script: `train.py`

### Base Model

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---------|------|
| epochs | 15 |
| batch_size | 12 |
| workers | 8 |
| patience | 15 |
| image size | 640 × 640 |

---

---

# 🧠 CNN Feature Extraction

All training images are resized to **640 × 640**.

The YOLO backbone CNN progressively downsamples:

640 × 640 → 20 × 20 feature grid


Each grid cell captures:

• facial orientation  
• head pose  
• hand movement  
• posture  
• body alignment  

These features are forwarded to the detection head for anchor-based regression.

---

---

# 🔄 Gradient Descent Optimization

During training, weights are updated using **back-propagation with gradient descent** to minimize the total YOLO loss:

\[
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{cls} L_{cls}
\]

Where:

• `L_box` → bounding box regression loss  
• `L_obj` → objectness confidence loss  
• `L_cls` → classification loss  

---

---

# 📐 YOLO Bounding Box Prediction Mathematics

For each anchor box, the network predicts:

(tx, ty, tw, th)


These are converted to image-space coordinates as:

### Center Coordinates

\[
b_x = \sigma(t_x) + c_x
\]

\[
b_y = \sigma(t_y) + c_y
\]

Where:

• `(c_x, c_y)` are grid-cell offsets  
• `σ` is the sigmoid function  

---

### Width & Height

\[
b_w = p_w \cdot e^{t_w}
\]

\[
b_h = p_h \cdot e^{t_h}
\]

Where:

• `(p_w, p_h)` are anchor dimensions  

---

### Final Confidence

\[
Score = P(object) \times P(class)
\]

---

---

# 3️⃣ Model Evaluation

### Script: `test_model.py`

### Configuration

split = test
workers = 8


---

## 📊 Metrics Computed

• Precision per activity  
• Recall  
• mAP@50  
• mAP@50-95  
• Confusion matrix  

---

---

# 4️⃣ Real-Time Inference & Tracking

### Script: `inference.py`

---

## 🎯 Detection Thresholds

| Parameter | Value |
|--------|------|
| Confidence Threshold | 0.18 |
| IoU Threshold | 0.35 |

---

---

# 🧭 Multi-Object Tracking

Tracker configuration:

bytetrack.yaml


Responsibilities:

• assigns persistent student IDs  
• handles occlusion  
• supports re-identification  

---

---

# 🎞 Temporal Smoothing

Predictions are stabilized using:

Window = 9 frames


Final class label = **majority vote** across window.

---

---

# 🔁 Re-Identification Logic

If a newly detected student appears within **90 pixels** of a previous track center:

➡ the original ID is reused.

---

---

# 📤 Inference Outputs

---

## 🎥 Annotated Video

outputs/output_inference.mp4


Displays:

• bounding boxes  
• student IDs  
• activity labels  
• confidence scores  

---

---

## 📄 Detection Log

outputs/detections_log.csv


Columns:

timestamp,
confidence,
student_id,
x1, y1, x2, y2,
activity


---

---

# 5️⃣ Post-Inference Analytics

### Script: `analyze.py`

---

## 📊 Statistical Analysis Performed

• mean confidence per class  
• standard deviation  
• frequency distribution  
• activity duration per student  
• detection reliability  
• class imbalance diagnostics  

---

---

# 🔁 End-to-End Pipeline Summary

Dataset A + Dataset B
↓
merge_datasets.py
↓
Merged Dataset + YAML
↓
train.py
↓
best.pt
↓ ↓
test_model.py inference.py
↓
output_inference.mp4 + detections_log.csv
↓
analyze.py


---

---

# 🚀 Applications

• classroom engagement monitoring  
• smart classroom analytics  
• academic research  
• behavioral modeling  
• automated attendance systems  

---

---
# 👤 Author

Nikhilesh Dubey


