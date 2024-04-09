from dotenv import dotenv_values

env_val = dotenv_values(".env")

runsetup = True
chunk_size = 600
chunk_overlap = 100
model_name = "gpt-3.5-turbo-0125" #replace with model 'gpt-4-0125-preview' for better reasoning, but it is slower so for testing I swapped out to 3.5
enableTTS = False
enableSTT = False
advanced_reasoning = False
userID = "999" #for testing 999 but we need to implement a user system
username = "John"
vectordb_path = "chromadb"
tts_api_key = env_val["TTS_API_KEY"] #optional for elevenlabs
openai_api_key = env_val["OPENAPI_KEY"] #put the API key, ive been using my own since profs stopped working
face_check_delay = 10
enableFaceRecognition = False