import face_recognition
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Data Collection:
# Collect a dataset of facial images with labeled emotions (e.g., happy, sad, neutral).
# Include diverse expressions, lighting conditions, and backgrounds.
# Preprocessing:
# Detect faces in the images using face detection algorithms (e.g., Viola-Jones).
# Normalize the images (resize, crop, and align) to a consistent format.
# Extract facial landmarks (e.g., eyes, nose, mouth) for feature extraction.

# Load the training data
train_data = []
train_labels = []
'''for emotion in os.listdir('diff image\images\train'):
    for img in os.listdir(f'diff image\images\train/{emotion}'):
        img_path = f'diff image\images\train/{emotion}/{img}'
'''
base_dir = 'face-reco/diff image/images/train'
for emotion in os.listdir(base_dir):
    for img in os.listdir(os.path.join(base_dir, emotion)):
        img_path = os.path.join(base_dir, emotion, img)
        img = face_recognition.load_image_file(img_path)
        img_encoding = face_recognition.face_encodings(img)[0]
        train_data.append(img_encoding)
        train_labels.append(emotion)
# Feature Extraction:
# Use deep learning models (e.g., Convolutional Neural Networks - CNNs) to extract features from facial landmarks.
# Common features include:
# Histogram of Oriented Gradients (HOG)
# Local Binary Patterns (LBP)
# Geometric features (distances between landmarks)

# Model Training:
# Split your dataset into training and validation sets.
# Train a CNN or other machine learning model on the features extracted from facial images.
# Use labeled emotions as target labels.

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the validation data
y_pred = model.predict(X_val)
print('Accuracy:', accuracy_score(y_val, y_pred))

# Real-time face recognition in live webcam feed:
# Find path of xml file containing haarcascade file
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt.xml"
# Load the cascade
faceCascade = cv2.CascadeClassifier(cascPathface);
# To capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the image
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face encoding
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (96, 96))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.array(face_img) / 255.0
        face_encoding = face_recognition.face_encodings(face_img)[0]

        # Predict the emotion
        emotion = model.predict([face_encoding])[0]

        # Display the predicted emotion
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display
    cv2.imshow('Video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()