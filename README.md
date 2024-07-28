# FRUIT-classification
# Fruit Classification

## Introduction
This project is based on a Wide and Deep Convolutional Neural Network (WDCNN) model for classifying pure water, kiwi, sugar oranges, and lemon juice.   The WDCNN model, with a 100% accuracy rate, cleverly distinguished the samples in just 15 seconds.   By leveraging the capabilities of deep learning, the  system can achieve rapid processing and classification of time series signals, thereby aiding in timely anomaly detection, prediction, and recognition.

## Environment Requirements
-Python 3.8 or higher version
-Torch
-Keras
-NumPy
-Pandas
-Scikit-learn

## Dataset
The dataset used in this project contains resistive signal data of various fruits.   The dataset has undergone data augmentation.


## Model Architecture
The WDCNN model combines the advantages of wide and deep convolutional neural networks, effectively extracting features from signals and performing classification.   The main architecture of the model includes:
Convolutional Layers: Used for feature extraction from signals.
Pooling Layers: Used to reduce the spatial dimensions of features while increasing invariance to shifts.
Fully Connected Layers: Used for the final classification decision.
## Usage
### Clone the Repository to Your Local Machine:
bash
git clone https://github.com/xiaoxiaocui007/FRUIT-classification.git/
### Enter the Project Directory:
bash
cd fruit-classification-wdcnn
### Run the Training Script:
bash
python train.py
### Run the Testing Script:
bash
python valid.py
### Model Training
The training script train.py will load the dataset, construct the WDCNN model, and train it on the training set.

### Model Evaluation
The testing script valid.py will load the trained model and evaluate its performance on the test set.

# Contribution
Contributions to this project are welcome.   If you have any suggestions or find issues, please submit a Pull Request or create an Issue.

# Contact
Email: [2574084240@qq.com]
