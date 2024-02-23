import configparser
import os

from openai import OpenAI
from langchain import PromptTemplate

if __name__ == "__main__":
    conf = configparser.ConfigParser()
    conf.read("config.properties")
    os.environ['OPENAI_API_KEY'] = conf['DEFAULT']['OPENAI_API_KEY']
    openai_client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    # Using the openai_client object above make a call to the OpenAI API
    # chat completions call with messages as list of dictionaries and model as the model name as paramters
    # and print the response
    stream = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Act as a professional in Python programming language."},
            {"role": "user", "content": "What is an async function in Python? Give an example with explination."},
        ],
        model="gpt-4-0125-preview",
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    # print(response.choices[0].message.content)
