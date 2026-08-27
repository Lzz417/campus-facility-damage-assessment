# MBDD2025: A Deep Learning-Oriented Dataset for Diverse Building Defect Detection from UAV Images

## 📂 Dataset Overview

**MBDD2025** is a large-scale dataset of building wall defects collected via UAV imagery. The dataset is designed for object detection and classification tasks in the field of intelligent building inspection. It includes a wide range of structural types and defect categories, making it suitable for generalization across various real-world scenarios.

- **Total images**: 14,471
- **Image resolution**: High-resolution UAV photos
- **Defect types (5)**:  
  - Crack  
  - Leakage  
  - Abscission  
  - Corrosion  
  - Bulge 

- **Building structure types (6)**:  
  - Steel structure  
  - Wooden structure
  - Masonry structure
  - Brick and wood structure
  - Brick concrete structure  
  - Reinforced concrete structure  

- **Annotation format**: VOC XML  
- **Compression format**: `.zip`

---

## 📁 Folder Structure
MBDD2025/
├── JPEGImages/ # Original UAV images
├── Annotations/ # VOC XML annotation files
├── Labels/ # Optional class label file (if available)
└── README.md # This file

---

## 🧾 Annotation Format

- Annotation files are in **PASCAL VOC XML** format.
- Each XML file corresponds to an image and contains:
  - Bounding boxes
  - Object class labels (defects only)
  - Image size and metadata

---

## 🚀 Usage

This dataset can be used to train object detection models such as:

- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [YOLOv5/YOLOv8](https://github.com/ultralytics/yolov5)

### Steps:

1. Download and unzip the dataset:

unzip MBDD2025.zip
cd MBDD2025

2. Use tools like [MMDetection](https://github.com/open-mmlab/mmdetection) or Ultralytics YOLO to train on the VOC format.

3. Make sure to convert labels if needed (e.g., to COCO or YOLO format).

---

## 📜 License

This dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
You are free to use, share, and adapt the dataset, provided appropriate credit is given.

---

## 📣 Citation

If you use **MBDD2025** in your research, please cite the dataset as follows:

@misc{MBDD2025,
title = {MBDD2025: A Deep Learning-Oriented Dataset for Diverse Building Defect Detection from UAV Images},
author = {Qikai Zha,Yiming Yao,Wenkang Zhang,Wanqi Ma},
year = {2025},
publisher = {Zenodo},
doi = {https://doi.org/10.5281/zenodo.15622584}
}

> 📌 *Replace `Your Name(s)` and `DOI` with the actual author(s) and DOI once the dataset is published on Zenodo.*

---

## 💬 Contact

If you have any questions or suggestions, feel free to open an issue or contact the dataset maintainer.
