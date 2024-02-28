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
Hi Tejaswi, Sumanth here.. the love of your life. I feel so many things about you love, and I can't wait to share them with you.

Every time you get adorably angry, it's hard for me not to smile. Your passion in every emotion, whether you're feeling playful, serious, or even cutely frustrated, makes me fall in love with you even more. It's like the universe crafted you to perfection for me. Your height, which you might think is less than average, is just perfect from my perspective. It puts your heart closer to mine, and honestly, I wouldn't have it any other way. You're not just my girlfriend; you're my daily dose of happiness and an embodiment of cuteness. Even when you're cutely angry, all I see is the woman whose spirit is as fiery as her heart is tender, and it just draws me in further. I've often thought about how the world perceives your size, but to me, you're larger than life. Every hug, every laugh, and even every mock glare you give me fills my life with immense love and joy. The world might see you as petite, but in my eyes, you're immense, filling every moment with endless wonder. Your cuteness knows no bounds, and your anger, as rare as it may be, comes with a charm that endears you to me even more. It stands as a testament to how lucky I am, constantly reminding me of the vibrant and full-of-emotion love that we share. The thought that sometimes, you might fret over being 'too this' or 'not enough that' never ceases to amaze me because, in every little thing you do, I see a perfect reflection of everything I've ever wanted. Truly, every time you furrow your brows in that cute anger, I'm reminded of how alive our love is, vibrant and full of emotion. How did I get so lucky to have you?"

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