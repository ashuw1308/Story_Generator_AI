# Bring in deps
import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")

os.environ['OPENAI_API_KEY'] = API_KEY

# App framework
st.title('🦜🔗 Detailed Story Generator')
prompt = st.text_input('Write details on the characters')

# Prompt templates
character_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a story title about {topic}'
)

theme_template = PromptTemplate(
    input_variables = ['title', 'imdb_search'],
    template='write me a story theme based on this title TITLE: {title} while leveraging this imdb search: {imdb_search} '
)

# Memory
character_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
theme_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9)
character_chain = LLMChain(llm=llm, prompt=character_template, verbose=True, output_key='title', memory=character_memory)
theme_chain = LLMChain(llm=llm, prompt=theme_template, verbose=True, output_key='script', memory=theme_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    title = character_chain.run(prompt)
    imdb_search = wiki.run(prompt)
    theme = theme_chain.run(title=title, imdb_search=imdb_search)

    st.write(title)
    st.write(theme)

    with st.expander('Character History'):
        st.info(character_memory.buffer)

    with st.expander('Theme History'):
        st.info(theme_memory.buffer)

    with st.expander('IMDB Search'):
        st.info(imdb_search)
