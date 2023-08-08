# Traffic-sign-classification

# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess your dataset here (similar to the preprocessing step provided earlier)

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model using your preprocessed data
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("traffic_sign_classifier.h5")

