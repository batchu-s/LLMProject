import streamlit as st
from openai import OpenAI
import configparser

conf = configparser.ConfigParser()
conf.read("config.properties")
api_key = conf["DEFAULT"]["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

st.title("💬 My Chatbot")
st.markdown("----")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0125-preview"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        response_stream = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True
        )
        response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # print(st.session_state.messages)