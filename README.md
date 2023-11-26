# Diabetes-Prediction-on-PIMA-dataset-using-CNN

## Overview

This project predicts whether a person has diabetes based on health-related features, using a Convolutional Neural Network (CNN).

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Asirwad/Diabetes-Prediction-on-PIMA-dataset-using-CNN.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the model:**
    ```bash
    python train.py
    ```

4. **Make predictions:**
    ```bash
    python predict.py
    ```

## Dataset

The dataset is the Pima Indian diabetes , containing features like pregnancies, glucose, blood pressure, etc.

## Model architecture
<div align="center">
  <table>
    <tr>
      <td><b><i>CNN Model</i></b></td>
    </tr>
    <tr>
      <td>
        <p align="center">
          <img src="app/models/model diagram.svg" alt="alt" width="100">
        </p>
      </td>
    </tr>
  </table>
</div>

## File Structure

- `train.py`: Script to train the CNN model.
- `predict.py`: Script to make predictions for individual cases.
- `model.h5`: Saved trained model.
- `scaler.joblib`: Saved StandardScaler for data preprocessing.

## Dependencies

- Python 3.x
- [TensorFlow](https://www.tensorflow.org/)
- Keras
- scikit-learn
- pandas
- numpy
