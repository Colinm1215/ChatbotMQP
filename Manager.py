import queue
import threading

import tools
from Interaction import Interaction
from realtime_testing import FaceRecognizer
import memory
import os
import config
import serial
import time
import Database as db

arduino_port = config.arduino_port
baud_rate = config.baud_rate
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(5)

message_queue = queue.Queue()
images_folder_path = "images"
face_recognizer = FaceRecognizer(images_folder_path, message_queue)
face_id_found = ""
current_interaction = None
lock = threading.Lock()


def command_motor(command):
    ser.write(command.encode())
    ser.flush()
    time.sleep(0.1)


def process_interaction(face_id, name):
    global current_interaction
    lock.acquire()
    try:
        current_interaction = Interaction(face_id, name)
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
                split_message = message.split("-")
                if (len(split_message) > 1):
                    face_id_in_message = split_message[0]
                    motor_instructions = split_message[1]

                    if face_id_found != face_id_in_message and face_id_found != "Error: No faces found":
                        face_id_found = face_id_in_message

                    print(f"face_id_found: {face_id_found}")
                    print(f"Sending command to motor : {motor_instructions}")
                    command_motor(motor_instructions)
                    if face_id_found != "" and current_interaction is None:
                        if face_id_found == "Error: No faces found":
                            continue  # Add Error Handling
                        elif face_id_found == "Error":
                            if config.enableTTS:
                                tools.elevenlabs(
                                    "Hello! My name is GompAI. I am sorry, but I do not recognize you - what is your name?")  # Convert the response to speech function for now
                            else:
                                print("Bot:",
                                      "Hello! My name is GompAI. I am sorry, but I do not recognize you - what is your name?")
                            if config.enableSTT:
                                user_input = tools.get_phrase()
                            else:
                                user_input = input("You: ")
                            id = db.generate_new_id()
                            db.insert_user(id, user_input)
                            threading.Thread(target=process_interaction, args=(id,user_input), daemon=True).start()
                        else:
                            threading.Thread(target=process_interaction, args=(face_id_found,db.search_name_by_id(face_id_found)), daemon=True).start()

    else:
        while True:
            threading.Thread(target=process_interaction, args=(face_id_found,), daemon=True).start()

except KeyboardInterrupt:
    face_recognizer.search = False
    face_recognizer.join()
    if current_interaction:
        current_interaction.stop()  # You may need to implement a stop method in Interaction class
    print("\nConversation ended.")
