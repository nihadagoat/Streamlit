import streamlit as st
import random
import time
from groq import Groq

#local env file
# import os
# from dotenv import load_dotenv
# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")

#get from streamlit secrets
API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=API_KEY)

client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"

response = client.chat.completions.create(
    model=MODEL,
    messages=st.session_state.messages

st.write("Streamlit loves LLMs! 🤖 [Build your own chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) in minutes, then make it powerful by adding images, dataframes, or even input widgets to the chat.")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = client.chat.completions.create(model=MODEL, messages=st.session_state.messages)

        assistant_response = response.choices[0].message.content
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
