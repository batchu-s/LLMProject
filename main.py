import requests
import boto3
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
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

def get_cohere_client():
    config = configparser.ConfigParser()
    config.read("config.properties")

    return CohereEmbeddings(model="embed-english-light-v2.0",cohere_api_key=config['DEFAULT']['COHERE_API_KEY'])

#--------------------------------------------------------------------------

if __name__ == "__main__":
    article_text = "Categories: IndiaBRICS nationsRepublics in the Commonwealth of NationsCountries and territories where English is an official languageFederal constitutional republicsFormer British colonies and protectorates in AsiaE7 nationsG15 nationsG20 nationsCountries and territories where Hindi is an official languageMember states of the Commonwealth of NationsMember states of the South Asian Association for Regional CooperationMember states of the United NationsSouth Asian countriesStates and territories established in 1947Countries in AsiaSocialist statesHidden categories: Pages using the Phonos extensionArticles with short descriptionShort description is different from WikidataFeatured articlesWikipedia indefinitely move-protected pagesWikipedia extended-confirmed-protected pagesUse Indian English from May 2020All Wikipedia articles written in Indian EnglishUse dmy dates from June 2023Articles containing Sanskrit-language textArticles containing Hindi-language textPages using infobox country or infobox former country with the symbol caption or type parametersPages using multiple image with auto scaled imagesArticles containing potentially dated statements from 2017All articles containing potentially dated statementsArticles containing potentially dated statements from 2010Articles containing potentially dated statements from 2009All articles with vague or ambiguous timeVague or ambiguous time from August 2023Articles containing potentially dated statements from 2020Articles containing potentially dated statements from 2012Pages using Sister project links with hidden wikidataPages using Sister project links with default searchArticles with Curlie linksArticles with FAST identifiersArticles with ISNI identifiersArticles with VIAF identifiersArticles with WorldCat Entities identifiersArticles with BIBSYS identifiersArticles with BNC identifiersArticles with BNE identifiersArticles with BNF identifiersArticles with BNFdata identifiersArticles with GND identifiersArticles with J9U identifiersArticles with LCCN identifiersArticles with Libris identifiersArticles with NDL identifiersArticles with NKC identifiersArticles with NLA identifiersArticles with PortugalA identifiersArticles with VcBA identifiersArticles with MusicBrainz area identifiersArticles with CINII identifiersArticles with Trove identifiersArticles with EMU identifiersArticles with HDS identifiersArticles with NARA identifiersArticles with SUDOC identifiersArticles with TDVİA identifiersCoordinates on WikidataArticles containing video clipsArticles containing image mapsArticles with accessibility problems"


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=10, length_function=len
    )

    all_split_texts = text_splitter.create_documents([article_text])

    all_split_texts_json = [json.dumps(jsonable_encoder(doc)) for doc in all_split_texts]
    embeddings = get_cohere_client().embed_documents(all_split_texts_json)

    print(len(embeddings[0]))