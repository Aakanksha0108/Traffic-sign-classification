import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("traffic_sign_classifier.h5")

cap = cv2.VideoCapture(0)  # Open the webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (32, 32))
    normalized_frame = resized_frame / 255.0

    prediction = model.predict(np.expand_dims(normalized_frame, axis=0))
    predicted_class = np.argmax(prediction)

    # Add logic here to trigger an alert based on the predicted_class

    cv2.imshow("Real-time Traffic Sign Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
