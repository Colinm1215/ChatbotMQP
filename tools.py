import shutil

from icalendar import Calendar
from datetime import datetime
from bs4 import BeautifulSoup
#from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from elevenlabs import generate, play, set_api_key
import config
import requests
import pyttsx3
from TTS.api import TTS
import torch
import elevenlabs
from playsound import playsound
import io
import speech_recognition as sr
import whisper
from tempfile import NamedTemporaryFile

device = "cuda" if torch.cuda.is_available() else "cpu"

if (config.enableTTS):
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)
    set_api_key(config.tts_api_key)

recorder = sr.Recognizer()
recorder.energy_threshold = 2500
recorder.dynamic_energy_threshold = False
source = sr.Microphone(sample_rate=16000)
audio_model = whisper.load_model("medium.en")
temp_file = NamedTemporaryFile().name

with source:
    recorder.adjust_for_ambient_noise(source)

def get_phrase():
    print("Listening")
    with source:
        audio_data = recorder.listen(source)

    wav_data = io.BytesIO(audio_data.get_wav_data())

    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())

    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()

    return text

def talk(speech):
    tts.tts_to_file(text=speech, file_path="./output.wav")
    playsound('./output.wav')

# Function for converting text to speech using pyttsx3
def text_to_speech(text, lang='en'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()


def elevenlabs(text):
    audio = generate(
        text=text,
        voice="Bella",
        model="eleven_multilingual_v2"
    )
    play(audio)

# Function to fetch and parse an iCalendar (.ics) file
def fetch_and_parse_ical(url):
    response = requests.get(url)
    if response.status_code == 200:
        gcal = Calendar.from_ical(response.content)
        events = []
        today = datetime.now().date()  # Get today's date
        for component in gcal.walk():
            if component.name == "VEVENT":
                start_dt = component.get('dtstart').dt
                if start_dt.date() == today:  # Check if the event is today
                    summary = component.get('summary')
                    end_dt = component.get('dtend').dt
                    events.append(f"{summary} - {start_dt} to {end_dt}")
        return events
    else:
        print(f"Failed to fetch iCalendar feed: HTTP {response.status_code}")

# Function to format a list of events into a conversational response

def format_event_response(events):
    if events:
        # Format the events into a conversational response
        response = "Here are the upcoming events: " + ", ".join(events)
    else:
        response = "I couldn't find any upcoming events at the moment."
    return response

def extract_and_chunk_text_from_pdf(pdfpath, chunk_size, overlap):  # Ada 2's optimal chunk size
    reader = PyPDFLoader(pdfpath)
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = chunk_size, chunk_overlap = overlap)
    extracted = reader.load_and_split(text_splitter)
    # Chunking the text for openai ada embbed model
    #chunks = text_splitter.create_documents(extracted)
    return extracted
    
def chunk_text_from_txt(txtpath, chunk_size, overlap):  # Ada 2's optimal chunk size
    loader = TextLoader(txtpath, encoding = 'UTF-8')
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = chunk_size, chunk_overlap = overlap, separators= '/f')
    extracted = loader.load_and_split(text_splitter)
    return extracted

#openai_embeddings = OpenAIEmbeddings(openai_api_key = config.openai_api_key)
#chroma_client = Chroma(embedding_function=openai_embeddings, persist_directory = config.vectordb_path)

#def create_and_store_embeddings(text, chroma_store): #dont use this function, will be replaced
    #for chunk in text:
       # embedding = openai_embeddings.embed(chunk)
       #  chroma_store.insert(embedding, chunk)

def update_config(file_path, key, new_value): #not being used right now
    updated_lines = []
    key_found = False

    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith(key):
                key_found = True
                # Preserve comments if present
                comment = ''
                if '#' in stripped_line:
                    value, comment = stripped_line.split('#', 1)
                    comment = '#' + comment
                updated_line = f"{key} = {new_value} {comment}\n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

    if not key_found:
        print(f"Key '{key}' not found in {file_path}")
        return

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

def reset_gompai():
    print("Are you sure you want to delete DB and reset config to default? (y/n)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["n"]:
            break
        if user_input.lower() in ["y"]:         
            config.chunk_size = 600
            config.chunk_overlap = 100
            config.model_name = "gpt-4-1106-preview"
            config.enableTTS = False
            config.advanced_reasoning = False
            config.userID = "999" #for testing 999
            config.vectordb_path = "chromadb"
                #config.openai_api_key = "" #put the API key
            shutil.rmtree("./")
            print("All settings in config reset, vector db deleted, re run program to initialize db again")
        print("Command not recognized, please try again with y/n")
        continue
        
#textmoment = chunk_text_from_txt(r'C:\Users\Eduardo\Documents\MQP Bot\txts\catalog.txt', 1024, 100)
#chroma_store = Chroma.from_documents(textmoment, openai_embeddings, client=chroma_client, collection_name ="docs_collection")


