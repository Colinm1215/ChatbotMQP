import os
import queue
import sqlite3

import config
from tools import Audible
from gompAI import gompAI, GompAgent
import Database as db


# if not self.message_queue.empty():
#     message = self.message_queue.get()
#     if self.face_id_found != message and self.face_id_found != "Error: No faces found":
#         self.face_id_found = message
# print(f"face_id_found : {self.face_id_found}")
# if self.face_id_found != "" and self.face_id_found != "Error: No faces found":

# def initialize_db(self):
#     if not os.path.exists('chromadb'):
#         print("'chromadb' directory not found. Initializing...")
#         memory.initialize_db()

# self.message_queue = message_queue
# self.face_recognizer = face_recognizer_class(images_folder_path, message_queue)
class Interaction:
    def __init__(self, id, name):
        self.id = id
        self.username = name
        self.running = True
        self.LLM = gompAI(name)
        self.audio = Audible()
        if config.enableTTS:
            self.audio.elevenlabs(f"Hello {self.username}!")
        else:
            print(f"Hello {self.username}")

    def get_user_input(self):
        if config.enableSTT:
            user_input = self.audio.get_phrase()
        else:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                self.running = False
        return user_input

    def process_messages(self):
        while self.running: # Come up with way to end conversation!
            user_input = self.get_user_input()
            if not self.running:
                break
            print(f"User Input : {user_input}")
            if user_input:
                response = self.LLM.get_chatbot_response(user_input)
                if config.enableTTS:
                    self.audio.elevenlabs(response)  # Convert the response to speech function for now
                else:
                    print("Bot:", response)
