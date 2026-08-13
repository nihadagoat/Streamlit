import re
import time
import math
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize API and config
API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Browser Buddy", page_icon="🤖", layout="wide")

# Persistent state initialization
if "users_db" not in st.session_state:
    st.session_state.users_db = {}

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "rag_documents" not in st.session_state:
    st.session_state.rag_documents = []


# --- AUTHENTICATION SCREEN ---
def show_auth_page():
    st.title("Browser Buddy 🤖")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Login", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()
    with col2:
        if st.button("Go to Sign Up", use_container_width=True):
            st.session_state.auth_mode = "signup"
            st.rerun()

    st.divider()

    if st.session_state.auth_mode == "login":
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Log In", type="primary", use_container_width=True):
            if login_user in st.session_state.users_db and st.session_state.users_db[login_user]["password"] == login_pass:
                st.session_state.current_user = login_user
                st.success(f"Welcome back, {login_user}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    else:
        st.subheader("Sign Up")
        signup_user = st.text_input("Username", key="signup_user")
        signup_pass = st.text_input("Password", type="password", key="signup_pass")
        
        if st.button("Create Account", type="primary", use_container_width=True):
            if not signup_user or not signup_pass:
                st.warning("Please fill out all fields.")
            elif signup_user in st.session_state.users_db:
                st.error("Username already exists. Try logging in.")
            else:
                st.session_state.users_db[signup_user] = {
                    "password": signup_pass,
                    "messages": [{"role": "assistant", "content": f"Welcome {signup_user}! Let's start chatting! 👇"}]
                }
                st.session_state.current_user = signup_user
                st.success("Account created successfully!")
                st.rerun()


if st.session_state.current_user is None:
    show_auth_page()
    st.stop()


# --- ISOLATED RAG ENGINE ---
class RAGSystem:
    def __init__(self, chunk_size=300, overlap=40):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def retrieve_context(self, query, documents, top_k=3):
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk_text(doc))
        
        if not chunks:
            return []

        vectorizer = TfidfVectorizer().fit(chunks + [query])
        chunk_vectors = vectorizer.transform(chunks)
        query_vector = vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [chunks[i] for i in top_indices if similarities[i] > 0.02]


rag_engine = RAGSystem()


def fetch_url_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return "\n".join(chunk for chunk in chunks if chunk)
    except Exception as e:
        return f"[Error fetching URL content: {e}]"


def extract_file_text(files):
    extracted_texts = []
    for file in files:
        file_content = f"Source File: {file.name}\n"
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    file_content += text + "\n"
        else:
            file_content += file.read().decode("utf-8", errors="ignore") + "\n"
        extracted_texts.append(file_content)
    return extracted_texts


# --- PAGE 1: STANDARD CHATBOT ---
def render_chatbot_page():
    user = st.session_state.current_user
    user_data = st.session_state.users_db[user]

    st.title(f"Browser Buddy Chat 🤖 ({user})")
    st.caption("Standalone LLM Chat Application")

    for message in user_data["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything..."):
        user_data["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            response = client.chat.completions.create(
                model=MODEL, 
                messages=user_data["messages"]
            )
            
            assistant_response = response.choices[0].message.content
            
            for chunk in assistant_response.split(" "):
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            
        user_data["messages"].append({"role": "assistant", "content": full_response})


# --- PAGE 2: SEPARATE RAG KNOWLEDGE BASE ---
def render_rag_page():
    st.title("RAG Knowledge Base & Query Engine 📚")
    st.caption("Upload documents or URLs to search and generate answers strictly from retrieved context.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Ingest Knowledge")
        uploaded_files = st.file_uploader(
            "Upload documents:", 
            type=["txt", "pdf", "py", "md", "json", "csv"], 
            accept_multiple_files=True
        )
        url_input = st.text_input("Or index a website URL:")
        
        if st.button("Add to Knowledge Base"):
            if uploaded_files:
                file_texts = extract_file_text(uploaded_files)
                st.session_state.rag_documents.extend(file_texts)
                st.success(f"Indexed {len(uploaded_files)} file(s)!")
            
            if url_input:
                url_text = fetch_url_content(url_input)
                st.session_state.rag_documents.append(url_text)
                st.success(f"Indexed content from {url_input}!")

        if st.button("Clear Knowledge Base"):
            st.session_state.rag_documents = []
            st.info("Knowledge base cleared.")

        st.info(f"Total documents currently indexed: **{len(st.session_state.rag_documents)}**")

    with col2:
        st.subheader("2. Query Knowledge Base")
        rag_query = st.text_area("Enter your question to query the indexed data:")
        
        if st.button("Run RAG Retrieval", type="primary"):
            if not st.session_state.rag_documents:
                st.warning("Please index at least one file or URL first.")
            elif not rag_query:
                st.warning("Please enter a question.")
            else:
                retrieved_chunks = rag_engine.retrieve_context(rag_query, st.session_state.rag_documents)
                
                if not retrieved_chunks:
                    st.error("No relevant content matched your query.")
                else:
                    st.subheader("Retrieved Chunks")
                    context_str = "\n\n---\n\n".join(retrieved_chunks)
                    st.code(context_str, language="text")

                    # Generate RAG response
                    st.subheader("RAG Answer")
                    rag_prompt = [
                        {"role": "system", "content": "You are a precise QA bot. Answer the question ONLY using the provided context chunks."},
                        {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {rag_query}"}
                    ]
                    
                    res = client.chat.completions.create(model=MODEL, messages=rag_prompt)
                    st.write(res.choices[0].message.content)


# --- NAVIGATION & SIDEBAR ---
with st.sidebar:
    st.write(f"Logged in as: **{st.session_state.current_user}**")
    if st.button("Log Out"):
        st.session_state.current_user = None
        st.rerun()
    st.divider()

page = st.navigation([
    st.Page(render_chatbot_page, title="Standard Chatbot", icon="💬"),
    st.Page(render_rag_page, title="RAG Knowledge Base", icon="📚"),
    st.Page(render_calculator_page, title="Scientific Calculator", icon="🧮")
])

page.run()
