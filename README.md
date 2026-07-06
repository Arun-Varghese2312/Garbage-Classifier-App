# Garbage Classification Using CNN

## Project Overview

This project uses a Convolutional Neural Network (CNN) to classify garbage images into 12 waste categories. The trained model is deployed using Streamlit for real-time image classification.

## Dataset

- Source: Kaggle Garbage Classification Dataset
- Number of Classes: 12
- Image Size: 180 × 180

Classes:
- Battery
- Biological
- Brown Glass
- Cardboard
- Clothes
- Green Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash
- White Glass

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Project Files

- `Garbage_Classification_using_CNN.ipynb` – Model development and training
- `garbage_classifier.keras` – Trained CNN model
- `class_names.json` – Class labels
- `app.py` – Streamlit application
- `test_model.py` – Model loading test
- `requirements.txt` – Python dependencies

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model Performance

- Training Accuracy: ~79.7%
- Validation Accuracy: ~61.8%

## Future Improvements

- Transfer Learning (EfficientNet/MobileNet)
- Data Augmentation
- Hyperparameter Optimization
- Larger Dataset

## Author

Arun Varghese
