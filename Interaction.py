import os
import queue
import config
import tools
import gompAI


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
    def __init__(self, username):
        self.username = username
        self.running = True
        self.LLM = gompAI(username)

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
