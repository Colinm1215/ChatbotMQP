import time

import cv2
from deepface import DeepFace
import os
import warnings
import pandas as pd
import threading
import queue

import config

warnings.filterwarnings('ignore')


class FaceRecognizer(threading.Thread):
    def __init__(self, images_folder_path, message_queue):
        threading.Thread.__init__(self)
        self.images_folder_path = images_folder_path
        self.message_queue = message_queue
        self.known_faces, self.labels = self.get_known_faces()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_recognized_face = None
        self.search = True

    def get_known_faces(self):
        """Load known faces and labels from a folder."""
        known_faces = []
        labels = []
        for filename in os.listdir(self.images_folder_path):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(self.images_folder_path, filename)
                image = cv2.imread(path)
                known_faces.append(image)
                labels.append(filename)
        return known_faces, labels

    def recognize_face(self, face):
        """Recognize a face against known faces."""
        for i, known_face in enumerate(self.known_faces):
            # Using DeepFace to verify faces
            result = DeepFace.verify(face, known_face, enforce_detection=False)
            if result["verified"]:
                return self.labels[i]
        return "Unknown"

    def run(self):
        last_check_time = time.time()
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        cached_results = None

        while self.search:
            ret, frame = cap.read()
            if not ret:
                self.message_queue.put("Error: Video capture failed")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) != 0:
                try:
                    db_path = os.path.abspath(self.images_folder_path)
                    cur_time = time.time()
                    if cached_results is None or cur_time - last_check_time >= config.face_check_delay:
                        last_check_time = time.time()
                        results = DeepFace.find(frame,
                                                db_path=db_path,
                                                model_name="VGG-Face",
                                                enforce_detection=False,
                                                silent=True)
                        cached_results = results

                    if cached_results:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                            for result_df in cached_results:
                                if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                                    for index, row in result_df.iterrows():
                                        identity = row['identity']
                                        recognized_face = os.path.basename(identity)
                                        if recognized_face != self.last_recognized_face:
                                            self.message_queue.put(f"{recognized_face.split('.', 1)[0]}")
                                            self.last_recognized_face = recognized_face
                                            break

                except Exception as e:
                    continue
                    #self.message_queue.put(f"Error: {e}")

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.search = False

        cap.release()
        cv2.destroyAllWindows()


# Usage example
if __name__ == "__main__":
    message_queue = queue.Queue()
    images_folder_path = "images"
    face_recognizer = FaceRecognizer(images_folder_path, message_queue)
    face_recognizer.start()

    # Main application loop
    try:
        while True:
            if not message_queue.empty():
                message = message_queue.get()
                #print(message)
            # Include other operations here
    except KeyboardInterrupt:
        face_recognizer.search = False
        face_recognizer.join()
