import cv2
from deepface import DeepFace
import os
import warnings
import pandas as pd

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def get_known_faces(folder_path):
    """Load known faces and labels from a folder."""
    known_faces = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            known_faces.append(image)
            labels.append(filename)
    return known_faces, labels

def recognize_face(face, known_faces, labels):
    """Recognize a face against known faces."""
    for i, known_face in enumerate(known_faces):
        # Using DeepFace to verify faces
        result = DeepFace.verify(face, known_face, enforce_detection=False)
        if result["verified"]:
            return labels[i]
    return "Unknown"

# Load known faces
known_faces, labels = get_known_faces("\images")

face_recognized = False
last_recognized_face = None

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV default) to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Using DeepFace to find a matching face
    try:
        # Assuming 'db_path' points to the folder with your known face images
        db_path = "C:\\Users\\fedpe\\OneDrive\\Desktop\\MPQTesting\\images"
        results = DeepFace.find(frame, db_path=db_path, model_name="VGG-Face", enforce_detection=False)

        if results:
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                for result_df in results:
                    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                        for index, row in result_df.iterrows():
                            identity = row['identity']
                            recognized_face = os.path.basename(identity)

                        # Check if the recognized face is different from the last recognized face
                            if recognized_face != last_recognized_face:
                                cv2.putText(frame, recognized_face, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                last_recognized_face = recognized_face
                                break

    except Exception as e:
        print(f"Error: {e}")

    # Display the frame
    cv2.imshow('Webcam', frame)
    
    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()