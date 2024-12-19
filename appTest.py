import os
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model
model = load_model('sign_language_recognizer_model.h5')

# Load label names (0-9 and A-Z)
LABELS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to detect hand landmarks, draw bounding boxes, and predict the sign in real-time
def detect_hand_and_predict_sign(frame):
    """Detect hand landmarks, draw bounding boxes, and predict the sign using the model in real-time."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75) as hands:
        results = hands.process(image_rgb)

        # If hand is detected, extract landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract hand landmarks (21 points with x, y, z)
                landmarks = []
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')

                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    # Append normalized coordinates (for model input)
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                    # Update bounding box coordinates
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)

                # Flatten the landmarks to pass them into the model
                landmarks = np.array(landmarks).flatten()

                # Predict the sign using the model
                prediction = model.predict(landmarks.reshape(1, -1))  # Reshape for model input
                predicted_class = np.argmax(prediction)
                predicted_label = LABELS[predicted_class]

                # Display the predicted label above the bounding box
                cv2.putText(frame, f'Sign: {predicted_label}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                return frame, predicted_label

    return frame, None


# Function to predict sign from uploaded image
def predict_from_uploaded_image(image):
    """Predict the sign from an uploaded image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.75) as hands:
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                # Flatten the landmarks to pass them into the model
                landmarks = np.array(landmarks).flatten()
                prediction = model.predict(landmarks.reshape(1, -1))
                predicted_class = np.argmax(prediction)
                predicted_label = LABELS[predicted_class]
                return predicted_label

    return "No hand detected"


# Streamlit UI
st.title('Sign Language Recognition')

# Add image to the right
col1, col2 = st.columns([2, 3]) # Adjust column ratio for layout

with col2:  # Display the image in the right column
    st.image('dataset1.png', caption='Datasets', use_column_width=True)

with col1:  # Main content in the left column
    # Option to select between image upload and real-time detection
    option = st.radio("Choose input method:", ("Upload Image", "Real-Time Detection"))

    if option == "Real-Time Detection":
        # Activate the webcam
        run = st.checkbox('Start Webcam')

        if run:
            # Open webcam
            cap = cv2.VideoCapture(0)

            # Create a placeholder for displaying video frames
            video_placeholder = st.empty()
            text_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video")
                    break

                # Detect hand landmarks, draw bounding boxes, and predict the sign
                frame_with_landmarks, predicted_sign = detect_hand_and_predict_sign(frame)

                # Update the video feed placeholder
                video_placeholder.image(frame_with_landmarks, channels="BGR")

                # Update the text with the predicted sign below the video
                if predicted_sign:
                    text_placeholder.success(f'Predicted Sign: {predicted_sign}')
                else:
                    text_placeholder.warning('No hand detected. Please raise your hand to make a gesture.')

                # Stop the webcam if the checkbox is unchecked
                if not run:
                    break

            cap.release()
        else:
            st.write("Webcam is stopped.")

    elif option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image of a hand gesture", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Predict the sign
            predicted_sign = predict_from_uploaded_image(image)

            # Display the uploaded image and the prediction
            st.image(image, channels="BGR", caption="Uploaded Image")
            st.success(f'Predicted Sign: {predicted_sign}')