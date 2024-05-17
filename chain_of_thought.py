import configparser
import os

from openai import OpenAI

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
            {"role": "system", "content": "Act as a professional Attorney."},
            {"role": "user", "content": "I am a Software Engineer with an H-1B visa. "
                                        "I also want to be a freelance web developer. "
                                        "What are the legal implications?"},
        ],
        model="gpt-4-0125-preview",
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    # print(response.choices[0].message.content)
