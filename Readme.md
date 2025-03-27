# Flower Classification Project

This project implements a flower classification model using TensorFlow 2.x and Keras. The project is designed in a modular and structured way, making it easy to understand, maintain, and extend.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project builds a convolutional neural network (CNN) for classifying flower images. It includes modules for:
- **Data Loading & Preprocessing:** Efficiently loads and splits image data into training and validation sets.
- **Model Building:** Defines a CNN model with integrated data augmentation.
- **Training & Evaluation:** Trains the model using callbacks (EarlyStopping and ModelCheckpoint), evaluates performance, and saves the best model.

## Project Structure


## Features

- **Modular Design:**  
  Code is separated into dedicated modules for data loading, model building, and training, enhancing readability and maintainability.
  
- **Data Augmentation:**  
  Integrated augmentation layers help improve the model’s generalization capabilities.

- **Efficient Data Pipeline:**  
  Uses TensorFlow’s caching and prefetching to optimize data loading during training.

- **Callbacks for Robust Training:**  
  Implements `EarlyStopping` to prevent overfitting and `ModelCheckpoint` to save the best-performing model automatically.

## Requirements

- Python 3.7 or later
- TensorFlow 2.x

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/rahul07rlk/Flower-Classification-main.git
   cd Flower-Classification-main
