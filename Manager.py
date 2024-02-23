import queue

from Interaction import Interaction
from realtime_testing import FaceRecognizer
import gompAI
import os


message_queue = queue.Queue()
images_folder_path = "images"
face_recognizer = FaceRecognizer(images_folder_path, message_queue)
face_recognizer.start()
face_id_found = ""
try:
    if not os.path.exists('chromadb'):
        print("'chromadb' directory not found. Initializing...")
        gompAI.memory.initialize_db()
    while True:
        if not message_queue.empty():
            message = message_queue.get()
            if face_id_found != message and face_id_found != "Error: No faces found":
                face_id_found = message
        print(f"face_id_found : {face_id_found}")
        if face_id_found != "":
            if face_id_found == "Error: No faces found":
                continue # Add Error Handling
            elif face_id_found == "Error: User Not Recognized":
                continue # Add Error Handling
            else:
                converstation = Interaction(face_id_found)
                converstation.process_messages()


except KeyboardInterrupt:
    face_recognizer.search = False
    face_recognizer.join()
    print("\nConversation ended.")