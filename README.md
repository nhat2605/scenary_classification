# Scenery Classification Model

## Overview
This repository contains a machine learning model for scenery classification, capable of identifying various types of landscapes such as buildings, forests, glaciers, mountains, seas, and streets. The model is built using TensorFlow and trained on a diverse dataset to ensure accurate classification.

This was a learning attempt on image classification, neural network and utilising GPU in training.

## Model Details
- **Framework**: TensorFlow
- **Model Type**: Convolutional Neural Network (CNN)
- **Classes**: Buildings, Forest, Glacier, Mountain, Sea, Street
- **Input Size**: 150x150 pixels

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow

### Installation
1. Clone the repository and Navigate to it:
```bash
https://github.com/nhat2605/scenary_classification.git
cd scenary_classification
```

2. Install the required packages: WIP

### Navigation
1. The scenery_classification.ipynb is a notebook file which demonstrates how the data was processed and used to train the model
2. The scenery_classify.py is a python script which helps users to easily use the model
3. The scenary_classification_model folder contains the trained model

### Usage
1. Run python3 scenery_classify.py
2. You will be prompted to insert a URL to an image. Please make sure this URL is accessible.
3. If the URL is valid, the model will return the possible result of what the image might be with a confidence percentage.

### Acknowledgement
1. The template of my training is from https://www.tensorflow.org/tutorials/images/classification

2. I do not own the data that was used to train the model. All data was retrieved from Kaggle.
