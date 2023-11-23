import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.arch import get_model

dataset_path = "dataset/diabetes.csv"
data = pd.read_csv(dataset_path)

# Split the data into features and labels
x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Reshape the data for CNN (add a channel dimension)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = get_model(input_shape=(x_train.shape[1], 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=150, batch_size=32, validation_data=(x_test, y_test))
model.save("models/model.h5")