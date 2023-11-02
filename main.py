import io
import speech_recognition as sr
import whisper
from tempfile import NamedTemporaryFile
import torch
from TTS.api import TTS
from playsound import playsound
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(device)
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
    print("Finished Listening, transcribing")

    wav_data = io.BytesIO(audio_data.get_wav_data())

    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())

    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()

    return text


def talk(speech):
    tts.tts_to_file(text=speech, file_path="./output.wav")
    playsound('./output.wav')


def main():
    while True:
        text = get_phrase()
        print("Sending to GPT")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": text}
            ]
        )
        print("Speaking")
        talk(completion.choices[0].message.content)


if __name__ == "__main__":
    main()