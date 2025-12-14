# Fashion Image Classifier

A deep learning-powered web application that classifies fashion items from images using a trained neural network model. Built with TensorFlow/Keras and deployed as an interactive web interface.

## 🎯 Project Overview

This project implements a fashion image classification system that can identify different types of clothing and accessories from images. The classifier is trained on the Fashion-MNIST dataset and can recognize 10 different categories of fashion items.

## 📦 Fashion Categories

The model can classify images into the following categories:

1. **T-shirt/top**
2. **Trouser**
3. **Pullover**
4. **Dress**
5. **Coat**
6. **Sandal**
7. **Shirt**
8. **Sneaker**
9. **Bag**
10. **Ankle boot**

## 🚀 Features

- **Image Upload**: Upload fashion item images for classification
- **Real-time Prediction**: Get instant classification results
- **Pre-trained Model**: Uses a pre-trained model (`fashion_mnist_model.h5`) for accurate predictions
- **User-Friendly Interface**: Simple and intuitive web interface
- **High Accuracy**: Trained on the Fashion-MNIST dataset with optimized architecture

## 🛠️ Technology Stack

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for model training and inference
- **Flask/Streamlit**: Web framework for the application interface
- **NumPy**: Numerical computations
- **PIL/OpenCV**: Image processing
- **Fashion-MNIST Dataset**: Training data source

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## 💻 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShreyasBorade30/fashion-image-classifier.git
   cd fashion-image-classifier
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000` (or the URL shown in the terminal)

3. **Upload an image**
   - Click on the upload button
   - Select a fashion item image from your device
   - Wait for the model to process and display the prediction

4. **View results**
   - The app will display the predicted category
   - Confidence scores for each class may also be shown

## 📁 Project Structure

```
fashion-image-classifier/
│
├── app.py                      # Main application file
├── fashion_mnist_model.h5      # Pre-trained model weights
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── uv.lock                    # Lock file for dependencies
├── .python-version            # Python version specification
├── .gitignore                 # Git ignore rules                 
│
└── assets/   
└── training/
├── fashion_mnist (1).ipynb    # training notebook

```

## 🧠 Model Architecture

The classifier uses a Convolutional Neural Network (CNN) architecture optimized for the Fashion-MNIST dataset:

- **Input Layer**: 28x28 grayscale images
- **Convolutional Layers**: Extract spatial features from images
- **Pooling Layers**: Reduce dimensionality and computational complexity
- **Dense Layers**: Perform final classification
- **Output Layer**: 10 classes with softmax activation

## 📊 Model Performance

The model is trained on the Fashion-MNIST dataset, which consists of:
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28x28 pixels (grayscale)

## 🔧 Configuration

The project uses `uv` for dependency management. You can modify the `pyproject.toml` file to adjust project settings and dependencies.


## 👤 Author

**Shreyas Borade**
- GitHub: [@ShreyasBorade30](https://github.com/ShreyasBorade30)


