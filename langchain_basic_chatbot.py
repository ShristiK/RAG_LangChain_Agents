# experimenting with creating a simple langchain chatbot
# used basic conversation chain to drive conversation where input output flow is fixed

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv

st.set_page_config(page_title='Virtual Reading Assistant')
st.markdown('#### :books:🧙LaLeviosa: Your Virtual Reading Assistant')
st.markdown(
    "<h8 style='text-align: right; color: green;'>*Tell us about your reading interest and we will suggest you the "
    "next book to read!!*</h8>",
    unsafe_allow_html=True)

# initializing empty list of messages, the list will contain messages to display
if "messages" not in st.session_state:
    st.session_state['messages'] = []

# creating the buffer memory for saving conv context to be passed to the llm
if "llm_memory" not in st.session_state:
    st.session_state['llm_memory'] = ConversationBufferWindowMemory(k=2)

# displaying old messages stored in session_state for reference
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# setting user's openai api key in os.environ var for use
openai_api_key = st.sidebar.text_input('Please enter your OpenAI API Key', type='password')
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    load_dotenv(override=True)


# function to call openai model to get response
def generate_response(user_question):
    template = "You are a friendly, literary assistant who helps users by suggesting them the next book they can read. " \
               "Users may share information about their genres of interest and the books they had enjoyed reading in the " \
               "past. In case user doesn't provide these information directly, it is your responsibility to ask probing " \
               "questions to understand the user's area of interest wrt books, genres and authors in an engaging way and " \
               "then recommend them a book to read. Make your answers crisp in not more than 200 words\n" \
               "{history}\n" \
               "Human: {input}\n" \
               "Assistant: "
    human_message_prompt = HumanMessagePromptTemplate.from_template(template)
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chatgpt_chain = ConversationChain(
        llm=ChatOpenAI(temperature=0),
        prompt=chat_prompt_template,
        verbose=True,
        memory=st.session_state['llm_memory']
    )
    # when using conversation chain no need to save the context explicitly as it is handled by the chain itself
    response = chatgpt_chain.predict(input=user_question)
    return response


# creating the chatbot interface and displaying
if prompt := st.chat_input('How can we help you today?'):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        print('Received user question, generating output')
        message_placeholder = st.empty()
        output = generate_response(prompt)
        message_placeholder.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
