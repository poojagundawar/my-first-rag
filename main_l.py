import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import base64

# Function to convert image to base64 to embed it in HTML
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Load configuration data
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    base_retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=base_retriever,
        return_source_documents=True
    )
    return chain

# Streamlit page configuration
st.set_page_config(
    page_title="Legal Doc Chat",
    page_icon="🏛️",
    layout="centered"
)

# Display background image using Streamlit's st.image
background_image_path = os.path.join(working_dir, "images", "legal1.jpg")
if os.path.exists(background_image_path):
    st.markdown(
        f"""
        <style>
        div[data-testid="stAppViewContainer"] {{
            background-image: url('data:image/jpeg;base64,{get_base64_image(background_image_path)}');
            background-size: cover;
            background-position: center;
            color: white;
        }}
        .st-emotion-cache-1flajlm {{
            color: #fff;
        }}
        div[data-testid="stChatInput"] textarea[data-testid="stChatInputTextArea"] {{
            margin: 10px !important;
        }}
        textarea:focus {{
            outline: none !important;
            border-radius: 5px;
        }}
        .custom-title {{
            color: #fafafa;
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            margin-top: -68px;
        }}
        div[data-testid="stChatInput"] > div {{
            border: 2px solid #000000 !important;
            border-radius: 8px;
            height: 60px;
        }}

         /* Align the main container to the right */
            div[data-testid="stAppViewBlockContainer"] {{
                # margin-left:600px;
                height: 80vh;
                overflow: scroll;
            }}

            /* Align the main container to the right */
            div[data-testid="stBottom"] {{
                #  margin-left:500px;
                #  width: 40%
            }}

             div[data-testid="stBottom"] .st-emotion-cache-128upt6 {{
                background-color: transparent;
             }}

            div[data-testid="stBottomBlockContainer"] {{
                padding-bottom: 20px;
                # margin-left: 150px;
            }}

            .st-emotion-cache-f4ro0r {{
                align-items: center;
            }}

        </style>
        """, unsafe_allow_html=True
    )
else:
    st.write("Background image not found!")

st.markdown('<h1 class="custom-title">🏛️ Legal Sense RAG (AI) System</h1>', unsafe_allow_html=True)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI...")

if user_input:
    # Append user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the conversational chain and append the response
    with st.chat_message("assistant"):
        response = st.session_state.conversational_chain({"query": user_input})
        assistant_response = response['result']
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
