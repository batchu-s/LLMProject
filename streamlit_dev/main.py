import streamlit as st
import pandas as pd
from mylogger import log
from openai import OpenAI
import configparser
from pathlib import Path

conf = configparser.ConfigParser()
conf.read("config.properties")
api_key = conf["DEFAULT"]["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)
speech_file_path = Path(__file__).parent / "speech.mp3"

st.title("💬 My Test")

description = """
Hi There!! This is Sumanth. I am a software engineer and I am working on a project called LLMProject.

"""

def audio():
    with client.audio.speech.with_streaming_response.create(
        input=description,
        model="tts-1-hd",
        voice="shimmer"
    ) as response:
        response.stream_to_file(speech_file_path)

audio()

audio_file = open('speech.mp3', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format="audio/mp3")