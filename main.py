import requests
import boto3
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
import pandas as pd
import configparser
from fastapi.encoders import jsonable_encoder
import json

def get_llm():
    bedrock_llm = BedrockEmbeddings(
        credentials_profile_name="sumanth",
        model_id="amazon.titan-e1t-medium",
        region_name="us-east-1",
    )
    return bedrock_llm

#--------------------------------------------------------------------------

def get_cohere_client():
    config = configparser.ConfigParser()
    config.read("config.properties")

    return CohereEmbeddings(model="embed-english-light-v2.0",cohere_api_key=config['DEFAULT']['COHERE_API_KEY'])

#--------------------------------------------------------------------------

def get_embedding(text: str) -> list:
    return get_cohere_client().embed_documents([text])[0]

#--------------------------------------------------------------------------

if __name__ == "__main__":
    ####################################################################
    # load documents
    ####################################################################
    # URL of the Wikipedia page to scrape
    url = 'https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom'

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the text on the page
    article_text = soup.get_text()

    ####################################################################
    # split text
    ####################################################################
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
    )


    all_split_texts = text_splitter.create_documents([article_text])

    # all_split_texts_json = [json.dumps(jsonable_encoder(doc)) for doc in all_split_texts]
    text_chunks = [item.page_content for item in all_split_texts]

    df = pd.DataFrame({'text_chunks': text_chunks[0:50]}) # only take first 50 chunks for now since cohere only allows 100 api calls per minute on trial keys
    
    
    df['cohere_embedding'] = df.text_chunks.apply(lambda x: get_embedding(x))
    print(df.head())