<<<<<<< HEAD
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot with OLLAMA"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistnat. Please respond to the user queires"),
        ("user","Question:{question}")
    ]
) 

def generate_response(question,engine,temperature,max_tokens):
    llm=OllamaLLM(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with Ollama")

#Drop down to select various Open AI models
llm = st.sidebar.selectbox("Select an AI Model",["mistral","phi3"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface for user input
st.write("Ask any question, Let me de-tangle your query😊")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
=======
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot with OLLAMA"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistnat. Please respond to the user queires"),
        ("user","Question:{question}")
    ]
) 

def generate_response(question,engine,temperature,max_tokens):
    llm=OllamaLLM(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")

#Drop down to select various Open AI models
llm = st.sidebar.selectbox("Select an Open AI Model",["mistral","phi3"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface for user input
st.write("Ask any question, Let me de-tangle your query😊")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
>>>>>>> 41e2725ae0dfc1d4a6fe69b62e5ad83b01b0bf9f
    st.write("Please provide the query")