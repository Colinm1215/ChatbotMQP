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

device = "cpu"




class Audible:

    def __init__(self):
        self.language = 'en'
        self.model_id = 'v3_en'
        self.sample_rate = 48000
        self.speaker = 'random'
        self.tts_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language=self.language,
                                      speaker=self.model_id)
        self.tts_model.to(device)

        if config.enableTTS:
            set_api_key(config.tts_api_key)

        self.stt_model, self.stt_decoder, self.stt_utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                           model='silero_stt',
                                                           language='en',
                                                           device=device)
        (self.read_batch, self.split_into_batches,
         self.read_audio_stt, self.prepare_model_input) = self.stt_utils

        self.vad_model, self.vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=True,
                                              onnx=False)

        (get_speech_timestamps,
         save_audio,
         read_audio_vad,
         VADIterator,
         collect_chunks) = self.vad_utils

        self.vad_iterator = VADIterator(self.vad_model)

        # Parameters
        self.FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.CHANNELS = 1  # Single channel for microphone
        self.RATE = 16000  # Sampling Rate
        self.CHUNK = 1024  # Number of frames per buffer

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
    def get_phrase(self):
        print("Listening...")
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True,
                            frames_per_buffer=self.CHUNK)
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
            conf = self.vad_model(torch.from_numpy(audio_float32), 16000)

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
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    print("written to file - transcribing")
                    test_files = glob('recorded_audio.wav')
                    batches = self.split_into_batches(test_files, batch_size=10)
                    print(batches)
                    input = self.prepare_model_input(self.read_batch(batches[0]),
                                                device=device)

                    output = self.stt_model(input)
                    for example in output:
                        text += self.stt_decoder(example.cpu())
                    frames.clear()
                    listening = False

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        return text

    def elevenlabs(self, text):
        audio = generate(
            text=text,
            model="eleven_multilingual_v2"
        )
        play(audio)

# def talk(speech):
#     audio = tts_model.apply_tts(text=speech,
#                                 speaker=speaker,
#                                 sample_rate=sample_rate)
#     audio_np = audio.numpy()
#
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
#         tmpfile_name = tmpfile.name
#         tmpfile.close()
#         AudioSegment(data=audio_np.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1).export(
#             tmpfile_name,
#             format="wav")
#         sd.play(audio_np, sample_rate)
#         sd.wait()


# Function for converting text to speech using pyttsx3
def text_to_speech(text, lang='en'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.say(text)
    engine.runAndWait()


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
