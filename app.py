import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PASSWORD = os.environ.get('PASSWORD')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.write("Please enter the password to access the chatbot.")
    password_input = st.text_input("Password", type="password")
    if st.button("Submit"):
        if password_input == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
else:
    # Move these functions inside the authenticated block
    @st.cache_resource
    def load_embeddings():
        return download_hugging_face_embeddings()

    @st.cache_resource
    def load_vector_store(_embeddings):
        return PineconeVectorStore.from_existing_index(
            index_name="medicalbot",
            embedding=_embeddings
        )

    # Load embeddings and vector store only when authenticated
    embeddings = load_embeddings()
    docsearch = load_vector_store(embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize LLM and chains
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Streamlit chat interface (main app content)
    st.title("Alevel Chatbot")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
