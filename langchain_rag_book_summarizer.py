from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, FAISS
import pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time

st.set_page_config(page_title='Document uploader')
st.markdown('#### :books:🧙LaLeviosa: Your Document Summarizer')
st.markdown(
    "<h8 style='text-align: right; color: green;'>*Share the pdf of the book you want to read and we will summarize "
    "it for you!!*</h8>",
    unsafe_allow_html=True)

openai_api_key = st.sidebar.text_input('Please enter your OpenAI API Key', type='password')
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    load_dotenv(override=True)

pinecone_api_key = st.sidebar.text_input('Please enter your Pinecone API Key', type='password')
if pinecone_api_key:
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
else:
    load_dotenv(override=True)

pinecone_env = st.sidebar.text_input('Please enter your Pinecone environment', type='password')
if pinecone_env:
    os.environ['PINECONE_ENV'] = pinecone_env
else:
    load_dotenv(override=True)


# initializing pinecone and creating a pinecone index
def initialize_pinecone_index(index_name='reading-list-summarizer'):
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV'])
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
    print('PineCone initialization done')
    return index_name


# loading documents from directory into a list of langchain document objects
def load_docs_from_directory(dir_path=''):
    documents = []
    for f_name in os.listdir(dir_path):
        if f_name.endswith(".pdf"):
            pdf_path = dir_path + '/' + f_name
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    print('Document loading done')
    return documents


# splitting documents and storing them in a vector store and returning a retriever for query
def get_vs_retriever_from_docs(doc_list, index_name):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(doc_list)
    print('Document splitting done')
    # uncomment if index already exists, else add documents to the index
    # vectordb = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())
    cnt = 0
    for doc in documents:
        if vectordb is None:
            vectordb = Pinecone.from_documents([doc], OpenAIEmbeddings(), index_name='reading-list-summarizer')
        else:
            vectordb.add_documents([doc])
        time.sleep(20)
        if cnt % 50 == 0:
            print(f'Count is {cnt}')
        cnt += 1
    print('Vector store and retriever creation done')
    return vectordb.as_retriever()


# returning response to user's question
def generate_response(user_question, vector_store_retriever):
    llm = ChatOpenAI(temperature=0)
    llm_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store_retriever,
                                                      # return_source_documents=True,
                                                      verbose=True,
                                                      memory=st.session_state['chat_history'])
    response = llm_chain({"question": user_question})
    return response['answer']


# streamlit ui components for getting file location and creating a vector store retriever from it
if "doc_upload" not in st.session_state:
    st.session_state['doc_upload'] = False
if "doc_retriever" not in st.session_state:
    st.session_state['doc_retriever'] = None
if "messages" not in st.session_state:
    st.session_state['messages'] = []

file_location_type = st.selectbox(label='Where is your file located?',
                                  options=['Local', ''],
                                  index=1)  # can be expanded to include cloud storage like s3
if file_location_type == 'Local':
    with st.form('my form'):
        dir_path = st.text_area('Please enter your file\'s directory address')
        submitted = st.form_submit_button('Submit')
        if submitted:
            doc_list = load_docs_from_directory(dir_path)
            index_name = initialize_pinecone_index()
            doc_retriever = get_vs_retriever_from_docs(doc_list, index_name='reading-list-summarizer')
            st.session_state['doc_retriever'] = doc_retriever
            st.session_state['doc_upload'] = True
            st.session_state['chat_history'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            print('State initialization done, waiting for user\'s input')
else:
    st.write('Please select a valid file location!')

if st.session_state['doc_upload']:
    # displaying old messages stored in session_state for reference
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# using conversation retrieval chain for querying based on the uploaded documents
if prompt := st.chat_input('How can we help you today?', disabled=not st.session_state['doc_upload']):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        output = generate_response(prompt, st.session_state['doc_retriever'])
        message_placeholder.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})
