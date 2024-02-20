from dotenv import dotenv_values

env_val = dotenv_values(".env")

runsetup = True
chunk_size = 600
chunk_overlap = 100
model_name = "gpt-4-1106-preview"
enableTTS = False
enableSTT = True
advanced_reasoning = False
userID = "999" #for testing 999 but we need to implement a user system
vectordb_path = "chromadb"
tts_api_key = env_val["TTS_API_KEY"] #optional for elevenlabs
openai_api_key = env_val["OPENAPI_KEY"] #put the API key, ive been using my own since profs stopped working
face_check_delay = 10