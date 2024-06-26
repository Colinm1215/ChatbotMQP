import time
import cv2
import serial
from deepface import DeepFace
import os
import warnings
import pandas as pd
import threading
import queue

import config

# import config

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
        self.cap_left = cv2.VideoCapture(0, cv2.CAP_ANY)
        self.cap_right = cv2.VideoCapture(1, cv2.CAP_ANY)

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

    def calculate_turn(self, center_x_left_camera, center_x_right_camera):
        steps_per_degree = config.steps_per_degree  # placeholder - set in config.py
        HFOV = config.HFOV  # placeholder - should also be set in config.py
        interocular_distance = config.interocular_distance  # placeholder - should also be set in config.py
        center_x_frame_right = int(self.cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)) / 2
        center_x_frame_left = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)) / 2

        center_x_avg = (center_x_left_camera + center_x_right_camera) / 2
        center_x_frame_avg = (center_x_frame_right + center_x_frame_left) / 2
        angle = ((center_x_avg - center_x_frame_avg) / interocular_distance) * (HFOV / 2)

        return angle * config.steps_per_degree

    def run(self):
        last_check_time = time.time()
        cached_results = None
        db_path = os.path.abspath(self.images_folder_path)
        center_x_left = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)) / 2
        center_x_right = int(self.cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)) / 2

        while self.search:
            ret_left, frame_left = self.cap_left.read()
            ret_right, frame_right = self.cap_right.read()
            if not ret_left or not ret_right:
                self.message_queue.put("Error: Video capture failed")
                break

            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            faces_left = self.face_cascade.detectMultiScale(gray_left, 1.1, minNeighbors=5, minSize=(40, 40))
            faces_right = self.face_cascade.detectMultiScale(gray_right, 1.1, minNeighbors=5, minSize=(40, 40))

            if len(faces_left) != 0:
                try:
                    cur_time = time.time()
                    if cached_results is None or cur_time - last_check_time >= 10:  # config.face_check_delay
                        last_check_time = time.time()
                        results = DeepFace.find(frame_left,
                                                db_path=db_path,
                                                model_name="VGG-Face",
                                                enforce_detection=False,
                                                silent=True)
                        cached_results = results
                    for (x, y, w, h) in faces_left:
                        cv2.rectangle(frame_left, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        center_x_left = x + (h / 2)
                        break

                except Exception as e:
                    continue
                    # self.message_queue.put(f"Error: {e}")

            if len(faces_right) != 0:
                try:
                    # cur_time = time.time()
                    # #if cached_results is None or cur_time - last_check_time >= 10:  # config.face_check_delay
                    # last_check_time = time.time()
                    for (x, y, w, h) in faces_right:
                        cv2.rectangle(frame_right, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        center_x_right = x + (h / 2)
                        break
                except Exception as e:
                    continue
                    # self.message_queue.put(f"Error: {e}")

            steps = self.calculate_turn(center_x_right, center_x_left)
            recognized_face = "Error"
            if cached_results:
                for result_df in cached_results:
                    #print(result_df)
                    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                        for index, row in result_df.iterrows():
                            identity = row['identity']
                            recognized_face = os.path.basename(identity).split('.', 1)[0]
                            #print(f"{recognized_face.split('.', 1)[0]},{steps},{direction}")
                            self.last_recognized_face = recognized_face
                            break

            if len(faces_left) != 0 and len(faces_right) != 0:
                if recognized_face == "Error":
                    results = DeepFace.find(frame_right,
                                            db_path=db_path,
                                            model_name="VGG-Face",
                                            enforce_detection=False,
                                            silent=True)
                    for result_df in results:
                        if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                            for index, row in result_df.iterrows():
                                identity = row['identity']
                                recognized_face = os.path.basename(identity).split('.', 1)[0]
                                # print(f"{recognized_face.split('.', 1)[0]},{steps},{direction}")
                                self.last_recognized_face = recognized_face
                                break
                direction = "N"
                if steps > 0:
                    direction = "R"
                elif steps < 0:
                    direction = "L"
                steps = round(abs(steps))
                str = "{" + f"{steps},{direction}" + "}"
                self.message_queue.put(f"{recognized_face.split('.', 1)[0]}-{str}")
                time.sleep(1)
            cv2.imshow('Left', frame_left)
            cv2.imshow('Right', frame_right)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.search = False

        self.cap_left.release()
        self.cap_right.release()
        cv2.destroyAllWindows()



# Usage example
if __name__ == "__main__":
    arduino_port = config.arduino_port
    baud_rate = config.baud_rate
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(5)
    message_queue = queue.Queue()
    images_folder_path = "images"
    face_recognizer = FaceRecognizer(images_folder_path, message_queue)
    face_recognizer.start()

    # Main application loop
    try:
        while True:
            if not message_queue.empty():
                message = message_queue.get()
                print("From Queue : " + message)
                split_message = message.split("-")
                if (len(split_message) > 1):
                    face_id_in_message = split_message[0]
                    motor_instructions = split_message[1]
                    print((motor_instructions + '\n').encode())
                    ser.write((motor_instructions + "\n").encode())
                    ser.flush()
                    time.sleep(0.1)
    except KeyboardInterrupt:
        face_recognizer.search = False
        face_recognizer.join()
