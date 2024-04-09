import queue

from Interaction import Interaction
from realtime_testing import FaceRecognizer
import memory
import os
import config

message_queue = queue.Queue()
images_folder_path = "images"
face_recognizer = FaceRecognizer(images_folder_path, message_queue)
face_id_found = ""
try:
    if not os.path.exists('chromadb'):
        print("'chromadb' directory not found. Initializing...")
        memory.initialize_db()  #I think this was just an artifact of moving stuff around but it wasnt being called either way, but idk
    if config.enableFaceRecognition:
        face_recognizer.start()
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
                    converstation = Interaction(face_id_found)
                    converstation.process_messages()
                else:
                    converstation = Interaction(face_id_found)
                    converstation.process_messages()
    else:
        while True:
            converstation = Interaction(config.userID)
            converstation.process_messages()
            # need to handle end of conversation, not sure if its best to do it in gompai or in interaction or here. maybe some saving context makes sense in gompai?


except KeyboardInterrupt:
    face_recognizer.search = False
    face_recognizer.join()
    print("\nConversation ended.")