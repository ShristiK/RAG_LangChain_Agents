# use of agent which is diff from chains as the steps aren't hard-coded here
# rather the agent takes actions based on input and intermediate results

import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents.conversational_chat.prompt import PREFIX

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
    st.session_state['llm_memory'] = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

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


def generate_response(user_question):
    llm = ChatOpenAI(temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=OpenAI(temperature=0), verbose=True)
    # search = DuckDuckGoSearchRun()
    search = SerpAPIWrapper()
    agent_tools = [
        #     Tool(
        #     name='DuckDuckGo Search',
        #     func=search.run,
        #     description="useful for when you need to answer questions about current events or the current state of the "
        #                 "world"
        # ),
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]
    system_msg_prompt = PREFIX + "\nAssistant is a friendly, literary assistant who helps users by suggesting them the next book they can read.\n " \
                                 "Users may share information about their genres of interest and the books they had enjoyed reading in the " \
                                 "past.\n In case user doesn't provide these information directly, it is the agent's responsibility to ask probing " \
                                 "questions to understand the user's area of interest in books, genres and authors in an engaging way and " \
                                 "then recommend them a book to read.\n Agent should also ensure that the responses to questions are not more than 200 words" \
                                 " and if the response is long, agent will first summarize the information and then respond back to the user"
    conv_agent = initialize_agent(agent_tools, llm, agent=AgentType.OPENAI_FUNCTIONS,
                                  # agent_kwargs={'system_message': system_msg_prompt},
                                  verbose=True,
                                  memory=st.session_state['llm_memory'])
    print('printing memory: ', conv_agent.memory.buffer)
    return conv_agent.run(input=user_question)


if prompt := st.chat_input('How can we help you today?'):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        output = generate_response(prompt)
        message_placeholder.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
