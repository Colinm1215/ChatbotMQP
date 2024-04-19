import shutil
import time
import wave
from datetime import datetime
from glob import glob

import numpy as np
import pyaudio
import pyttsx3
import requests
import torch
from elevenlabs import generate, play, set_api_key
from icalendar import Calendar
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config
import sounddevice as sd
from pydub import AudioSegment
import tempfile

device = "cuda" if torch.cuda.is_available() else "cpu"

language = 'en'
model_id = 'v3_en'
sample_rate = 48000
speaker = 'random'

tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language=language,
                              speaker=model_id)
tts_model.to(device)

if config.enableTTS:
    set_api_key(config.tts_api_key)

stt_model, stt_decoder, stt_utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                   model='silero_stt',
                                                   language='en',
                                                   device=device)
(read_batch, split_into_batches,
 read_audio_stt, prepare_model_input) = stt_utils

vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True,
                                      onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio_vad,
 VADIterator,
 collect_chunks) = vad_utils

vad_iterator = VADIterator(vad_model)

# Parameters
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for microphone
RATE = 16000  # Sampling Rate
CHUNK = 1024  # Number of frames per buffer

# Initialize PyAudio
audio = pyaudio.PyAudio()


def get_phrase():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    text = ""
    frames = []
    recording = False
    last_voice_time = time.time()
    listening = True
    while listening:
        audio_chunk = stream.read(1536)
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        # Following code chunk credited to Alexander Veysov
        abs_max = np.abs(audio_int16).max()
        sound = audio_int16.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
        audio_float32 = sound.squeeze()  # depends on the use case
        # End of code chunk
        conf = vad_model(torch.from_numpy(audio_float32), 16000)

        current_time = time.time()
        if conf[0][0] > 0.2:
            recording = True
            last_voice_time = current_time
        elif current_time - last_voice_time > 0.5:
            recording = False

        if recording:
            frames.append(audio_chunk)
        else:
            if len(frames) > 0:
                print("ATtempting to write to file")
                wav_file_path = "recorded_audio.wav"  # Define your WAV file name
                wf = wave.open(wav_file_path, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                print("written to file - transcribing")
                test_files = glob('recorded_audio.wav')
                batches = split_into_batches(test_files, batch_size=10)
                input = prepare_model_input(read_batch(batches[0]),
                                            device=device)

                output = stt_model(input)
                for example in output:
                    text += stt_decoder(example.cpu())
                frames.clear()
                listening = False

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    return text


def talk(speech):
    audio = tts_model.apply_tts(text=speech,
                                speaker=speaker,
                                sample_rate=sample_rate)
    audio_np = audio.numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name
        tmpfile.close()
        AudioSegment(data=audio_np.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1).export(
            tmpfile_name,
            format="wav")
        sd.play(audio_np, sample_rate)
        sd.wait()


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    extracted = reader.load_and_split(text_splitter)
    # Chunking the text for openai ada embbed model
    # chunks = text_splitter.create_documents(extracted)
    return extracted


def chunk_text_from_txt(txtpath, chunk_size, overlap):  # Ada 2's optimal chunk size
    loader = TextLoader(txtpath, encoding='UTF-8')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators='/f')
    extracted = loader.load_and_split(text_splitter)
    return extracted


# openai_embeddings = OpenAIEmbeddings(openai_api_key = config.openai_api_key)
# chroma_client = Chroma(embedding_function=openai_embeddings, persist_directory = config.vectordb_path)

# def create_and_store_embeddings(text, chroma_store): #dont use this function, will be replaced
# for chunk in text:
# embedding = openai_embeddings.embed(chunk)
#  chroma_store.insert(embedding, chunk)

def update_config(file_path, key, new_value):  # not being used right now
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
            config.userID = "999"  # for testing 999
            config.vectordb_path = "chromadb"
            # config.openai_api_key = "" #put the API key
            shutil.rmtree("./")
            print("All settings in config reset, vector db deleted, re run program to initialize db again")
        print("Command not recognized, please try again with y/n")
        continue

# textmoment = chunk_text_from_txt(r'C:\Users\Eduardo\Documents\MQP Bot\txts\catalog.txt', 1024, 100)
# chroma_store = Chroma.from_documents(textmoment, openai_embeddings, client=chroma_client, collection_name ="docs_collection")
