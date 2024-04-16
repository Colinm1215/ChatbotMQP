import queue
import threading
from Interaction import Interaction
from realtime_testing import FaceRecognizer
import memory
import os
import config
import serial
import time

arduino_port = config.arduino_port
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)

message_queue = queue.Queue()
images_folder_path = "images"
face_recognizer = FaceRecognizer(images_folder_path, message_queue)
face_id_found = ""
current_interaction = None
lock = threading.Lock()


def command_motor(command):
    ser.write(command.encode())
    time.sleep(0.1)


def process_interaction(face_id):
    global current_interaction
    lock.acquire()
    try:
        current_interaction = Interaction(face_id)
        current_interaction.process_messages()
    finally:
        current_interaction = None  # Reset current_interaction
        lock.release()


try:
    if not os.path.exists('chromadb'):
        print("'chromadb' directory not found. Initializing...")
        memory.initialize_db()

    if config.enableFaceRecognition:
        face_recognizer.start()

        while True:
            if not message_queue.empty():
                message = message_queue.get()
                split_message = message.split(",")
                face_id_in_message = split_message[0]
                motor_instructions = split_message[1]

                if face_id_found != face_id_in_message and face_id_found != "Error: No faces found":
                    face_id_found = face_id_in_message

                print(f"face_id_found: {face_id_found}")
                print(f"Sending command to motor : {motor_instructions}")
                command_motor(motor_instructions)
                if face_id_found != "":
                    if face_id_found == "Error: No faces found":
                        continue  # Add Error Handling
                    elif face_id_found == "Error: User Not Recognized":
                        threading.Thread(target=process_interaction, args=(face_id_found,), daemon=True).start()
                    else:
                        threading.Thread(target=process_interaction, args=(face_id_found,), daemon=True).start()

    else:
        while True:
            threading.Thread(target=process_interaction, args=(config.userID,), daemon=True).start()

except KeyboardInterrupt:
    face_recognizer.search = False
    face_recognizer.join()
    if current_interaction:
        current_interaction.stop()  # You may need to implement a stop method in Interaction class
    print("\nConversation ended.")
