# Diabetes Prediction Project

## Overview

This project predicts whether a person has diabetes based on health-related features, using a Convolutional Neural Network (CNN).

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/diabetes-prediction.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the model:**
    ```bash
    python train_model.py
    ```

4. **Make predictions:**
    ```bash
    python predict.py
    ```

## Dataset

The dataset is the Pima Indian diabetes dataset, containing features like pregnancies, glucose, blood pressure, etc.

## Model Details

The CNN model is built using TensorFlow and Keras, with convolutional layers for feature extraction.

## File Structure

- `train_model.py`: Script to train the CNN model.
- `predict.py`: Script to make predictions for individual cases.
- `diabetes_model.h5`: Saved trained model.
- `scaler.joblib`: Saved StandardScaler for data preprocessing.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- scikit-learn
- pandas
- numpy

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
