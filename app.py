import streamlit as st
from src.helper import download_hugging_face_embeddings, retrieve_menu
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

# Limit session history
MAX_HISTORY_LENGTH = 3  # Keep the last 3 turns
# Menu configuration data
MENU_DATA = retrieve_menu()

TEST = False

SHOW_RETRIEVED_RECORDS = False if not TEST else True

# Custom CSS to reduce spacing between items and adjust layout
st.markdown("""
    <style>
        .sidebar .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        ul {
            margin-bottom: 0.3rem; /* Reduce spacing between list items */
        }
    </style>
""", unsafe_allow_html=True)


# Authentication check
if "authenticated" not in st.session_state:
    if TEST:
        st.session_state.authenticated = True
    else:
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
    @st.cache_data
    def load_menu():
        return retrieve_menu()

    MENU_DATA = load_menu()

    @st.cache_resource
    def load_embeddings():
        return download_hugging_face_embeddings()

    @st.cache_resource
    def load_vector_store(_embeddings, namespace=None):
        return PineconeVectorStore.from_existing_index(
            index_name="alevelbot",
            embedding=_embeddings,
            namespace=namespace
        )

    # Initialize LLM and chains
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=2048)
    # Load embeddings and vector store only when authenticated
    embeddings = load_embeddings()

    # Sidebar menus
    st.sidebar.title("Menu")

    # Unit selection dropdown in sidebar
    selected_unit = st.sidebar.selectbox(
        "Select Unit",
        options=[item["namespace-key"] for item in MENU_DATA],
        format_func=lambda key: next((item["name"] for item in MENU_DATA if item["namespace-key"] == key), key),
        index=None  # Default to the first unit
    )
    
    # Topic selection and display logic
    selected_topic = None
    current_unit = next((item for item in MENU_DATA if item["namespace-key"] == selected_unit), None)
    
    if current_unit and current_unit.get("topics"):
        topic_options = ["All Topics"] + [topic["name"] for topic in current_unit["topics"]]
        selected_topic = st.sidebar.selectbox(
            "Select Topic",
            options=topic_options,
            index=0  # Default to "All Topics"
        )

        # Display topics and subtopics in sidebar with reduced spacing
        st.sidebar.write("### Topics Overview")
        if selected_topic == "All Topics":
            for topic in current_unit["topics"]:
                st.sidebar.markdown(f"- **{topic['name']}**", unsafe_allow_html=True)
        else:
            selected_topic_data = next((t for t in current_unit["topics"] if t["name"] == selected_topic), None)
            if selected_topic_data:
                st.sidebar.markdown(f"**{selected_topic_data['name']}**", unsafe_allow_html=True)
                for subtopic in selected_topic_data.get("subtopics", []):
                    st.sidebar.markdown(f"- {subtopic}", unsafe_allow_html=True)

    # Only initialize vector store and chain if unit is selected
    if selected_unit:
        docsearch = load_vector_store(embeddings, selected_unit)
        
        # Configure retriever with filters
        search_kwargs = {"k": 5, "score_threshold": 0.6}

        if selected_topic and selected_topic != "All Topics":
            search_kwargs["filter"] = {"topic": selected_topic.title()}
        
        # retriever = docsearch.as_retriever(
        #     search_type="similarity", 
        #     search_kwargs=search_kwargs)

        retriever = docsearch.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs=search_kwargs)


        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # Streamlit chat interface (main app content)
        st.title("A-level Chatbot")



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

            # Trim history
            if len(st.session_state.messages) > MAX_HISTORY_LENGTH * 2:
                st.session_state.messages = st.session_state.messages[-MAX_HISTORY_LENGTH * 2:]

            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_input + " Include the videos, images and referencing URLs if they exist in retrieved context and are relevant!"})
                answer = response["answer"]

                retrieved_docs = response["context"]

                number_docs = len(retrieved_docs)

                if SHOW_RETRIEVED_RECORDS:

                    if len(retrieved_docs) == 0:
                        st.markdown(f"No Docs Retrieved")

                    # Display retrieved documents
                    st.subheader("Retrieved Documents:")
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Document {i}:**")
                        st.markdown(doc.page_content)
                        st.markdown("Metadata:")
                        st.json(doc.metadata)
                        st.markdown("---")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer, unsafe_allow_html=True)
                st.markdown(f"Found {number_docs} Relevant Documents")
