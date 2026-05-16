# Car Classification AI (Google Colab Version)

This project provides an automated vehicle classification system using YOLOv8. It is optimized for Google Colab to avoid local environment issues.

## 🚀 Getting Started with Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `Car_Classifier_Colab.ipynb`.
3. Follow the steps in the notebook to:
   - Install dependencies.
   - Run the classifier on a real dataset.
   - View results and performance metrics.

## 📂 Project Structure

- `car_classifier.py`: The main Python script (Computer Vision logic).
- `Car_Classifier_Colab.ipynb`: Optimized notebook for cloud testing.
- `requirements.txt`: Python dependencies.

## 🚗 Features

- **YOLOv8 Detection**: High-accuracy vehicle detection.
- **Sub-type Classification**: Distinguishes between sedans, SUVs, trucks, vans, etc.
- **Image Quality Check**: Detects brightness and blur issues.
- **Parking Recommendations**: Suggests spot types based on vehicle size.

## 📊 Testing Accuracy

To test accuracy, we use a real-world dataset of car images. The Colab notebook automatically fetches test samples to demonstrate the system's performance.
