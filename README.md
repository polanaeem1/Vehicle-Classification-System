# Car Classification AI

This project provides an automated vehicle classification system using YOLOv8 and multiple other architectures. It is optimized for both Google Colab and Kaggle to accommodate easy training, benchmarking, and testing.


## 🏆 Kaggle Benchmarking Framework

We provide a comprehensive Kaggle notebook (`car-benchmark-framework.ipynb`) for augmenting data, training, and evaluating multiple models on a large scale (~200k images).

1. Upload `car-benchmark-framework.ipynb` to [Kaggle](https://www.kaggle.com/).
2. Enable **GPU T4 x2** or **P100** and toggle **Internet ON**.
3. (Optional) Upload your local datasets (`Datasets/`, `Car-Bike-Dataset/`).
4. Click **Run All** to:
   - Download the Stanford Cars dataset and merge it with local data.
   - Apply heavy data augmentation (Albumentations) to balance 7 classes (compact, motorcycle, sedan, suv, truck, van).
   - Train and benchmark YOLOv8, ResNet18, EfficientNet-B0, and MobileNetV3.
   - Output confusion matrices and detailed Top-1 / Top-5 accuracy metrics.

## 📂 Project Structure

- `car_classifier.py`: The main Python script (Computer Vision logic).
- `Car_Classifier_Colab.ipynb`: Optimized notebook for Google Colab testing.
- `car-benchmark-framework.ipynb`: Kaggle notebook for dataset generation and model benchmarking.
- `requirements.txt`: Python dependencies.

## 🚗 Features

- **YOLOv8 Detection**: High-accuracy vehicle detection.
- **Model Benchmarking**: Compares YOLOv8 against PyTorch models (ResNet18, EfficientNet-B0, MobileNetV3).
- **Sub-type Classification**: Distinguishes between sedans, SUVs, trucks, vans, motorcycles, and compact cars.
- **Large-Scale Data Augmentation**: Robust image transformations to reach a balanced 200k+ dataset size.

## 📊 Testing Accuracy

To test accuracy, we use a real-world dataset of car images. The Colab notebook easily fetches test samples to demonstrate the system's performance, while the Kaggle notebook runs an exhaustive, large-scale training and evaluation pipeline across multiple architectures.
