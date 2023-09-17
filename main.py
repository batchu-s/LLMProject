import requests
import boto3
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
import pandas as pd
import configparser
from fastapi.encoders import jsonable_encoder
import time
import json
import numpy as np

def get_llm():
    """This function returns the amazon Bedrock client

    Returns:
        _type_: Bedrock client
    """
    bedrock_llm = BedrockEmbeddings(
        credentials_profile_name="sumanth",
        model_id="amazon.titan-e1t-medium",
        region_name="us-east-1",
    )
    return bedrock_llm

#--------------------------------------------------------------------------

def get_cohere_client():
    """This function returns the cohere client

    Returns:
        _type_: cohere client
    """
    config = configparser.ConfigParser()
    config.read("config.properties")

    return CohereEmbeddings(
        model="embed-english-light-v2.0",
        cohere_api_key=config['DEFAULT']['COHERE_API_KEY']
    )

#--------------------------------------------------------------------------

def get_embedding(text: str) -> list:
    """The function returns the embedding of the text frpom the cohere client

    Args:
        text (str): The text chunk from the dataframe

    Returns:
        list: Embedding vector of the text chunk
    """
    time.sleep(0.6)
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

    df = pd.DataFrame({'text_chunks': text_chunks}) # only take first 50 chunks for now since cohere only allows 100 api calls per minute on trial keys
    
    
    df['cohere_embedding'] = df.text_chunks.apply(lambda x: get_embedding(x))
    # print(df.head())

    # Calculate embeddings for the user's question.
    users_question = "Who is the prime minister of the United Kingdom?"

    question_embedding = get_embedding(users_question)

    # create a list to store calculated cosine similarity
    cosine_similarity = []

    for index, row in df.iterrows():
       A = row.cohere_embedding
       B = question_embedding
       
       cosine = np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

       cosine_similarity.append(cosine)

    df['cosine_similarity'] = cosine_similarity
    df.sort_values(by=['cosine_similarity'], ascending=False)
    print(df.head())