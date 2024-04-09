import os
import queue
import config
import tools
from gompAI import gompAI
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
    def __init__(self, id):
        self.id = id
        if self.id == "":
            if config.enableTTS:
                tools.elevenlabs("Hello! My name is GompAI. I am sorry, but I do not recognize you - what is your name?")  # Convert the response to speech function for now
            else:
                print("Bot:", "Hello! My name is GompAI. I am sorry, but I do not recognize you - what is your name?")
            user_input = self.get_user_input() # figure out some way to strip only the name from user_input - maybe some custom chatGPT prompt or NLP model or even clever Regex
            db.insert_user(db.generate_new_id(), user_input)

        self.username = db.search_name_by_id(id)
        self.running = True
        self.LLM = gompAI(self.username)

    def get_user_input(self):
        if config.enableSTT:
            user_input = tools.get_phrase()
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
                    tools.elevenlabs(response)  # Convert the response to speech function for now
                else:
                    print("Bot:", response)
