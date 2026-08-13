import re
import math
import time
import json
import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "users_db.json"


def load_users():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_users(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=API_KEY)
MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Browser Buddy", page_icon="🤖", layout="wide")

if "users_db" not in st.session_state:
    st.session_state.users_db = load_users()

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"

if "rag_documents" not in st.session_state:
    st.session_state.rag_documents = []

if "calc_input" not in st.session_state:
    st.session_state.calc_input = ""


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
            st.session_state.users_db = load_users()
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
            st.session_state.users_db = load_users()
            if not signup_user or not signup_pass:
                st.warning("Please fill out all fields.")
            elif signup_user in st.session_state.users_db:
                st.error("Username already exists. Try logging in.")
            else:
                st.session_state.users_db[signup_user] = {
                    "password": signup_pass,
                    "messages": [{"role": "assistant", "content": f"Welcome {signup_user}! Let's start chatting! 👇"}]
                }
                save_users(st.session_state.users_db)
                st.session_state.current_user = signup_user
                st.success("Account created and saved successfully!")
                st.rerun()


if st.session_state.current_user is None:
    show_auth_page()
    st.stop()


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
        save_users(st.session_state.users_db)
        
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
        save_users(st.session_state.users_db)


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

                    st.subheader("RAG Answer")
                    rag_prompt = [
                        {"role": "system", "content": "You are a precise QA bot. Answer the question ONLY using the provided context chunks."},
                        {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {rag_query}"}
                    ]
                    
                    res = client.chat.completions.create(model=MODEL, messages=rag_prompt)
                    st.write(res.choices[0].message.content)


# --- PAGE 3: CALCULATOR WITH MATHEMATICAL KEYBOARD ---
def render_calculator_page():
    st.title("Scientific Calculator 🧮")
    st.caption("Enter math expressions via typing or using the mathematical keyboard buttons below.")

    expr = st.text_input("Expression:", value=st.session_state.calc_input, key="calc_display")
    st.session_state.calc_input = expr

    def append_val(val):
        st.session_state.calc_input += str(val)
  

    def clear_val():
        st.session_state.calc_input = ""
    

    def backspace():
        st.session_state.calc_input = st.session_state.calc_input[:-1]
        

    st.subheader("Mathematical Keyboard")
    
    r1_col1, r1_col2, r1_col3, r1_col4, r1_col5, r1_col6 = st.columns(6)
    r1_col1.button("sin", on_click=append_val, args=("sin(",), use_container_width=True)
    r1_col2.button("cos", on_click=append_val, args=("cos(",), use_container_width=True)
    r1_col3.button("tan", on_click=append_val, args=("tan(",), use_container_width=True)
    r1_col4.button("log", on_click=append_val, args=("log(",), use_container_width=True)
    r1_col5.button("ln", on_click=append_val, args=("ln(",), use_container_width=True)
    r1_col6.button("√", on_click=append_val, args=("sqrt(",), use_container_width=True)

    r2_col1, r2_col2, r2_col3, r2_col4, r2_col5, r2_col6 = st.columns(6)
    r2_col1.button("x²", on_click=append_val, args=("**2",), use_container_width=True)
    r2_col2.button("xⁿ", on_click=append_val, args=("**",), use_container_width=True)
    r2_col3.button("π", on_click=append_val, args=("pi",), use_container_width=True)
    r2_col4.button("e", on_click=append_val, args=("e",), use_container_width=True)
    r2_col5.button("(", on_click=append_val, args=("(",), use_container_width=True)
    r2_col6.button(")", on_click=append_val, args=(")",), use_container_width=True)

    r3_col1, r3_col2, r3_col3, r3_col4, r3_col5, r3_col6 = st.columns(6)
    r3_col1.button("7", on_click=append_val, args=("7",), use_container_width=True)
    r3_col2.button("8", on_click=append_val, args=("8",), use_container_width=True)
    r3_col3.button("9", on_click=append_val, args=("9",), use_container_width=True)
    r3_col4.button("÷", on_click=append_val, args=("/",), use_container_width=True)
    r3_col5.button("DEL", on_click=backspace, use_container_width=True)
    r3_col6.button("CLEAR", on_click=clear_val, use_container_width=True)

    r4_col1, r4_col2, r4_col3, r4_col4, r4_col5, r4_col6 = st.columns(6)
    r4_col1.button("4", on_click=append_val, args=("4",), use_container_width=True)
    r4_col2.button("5", on_click=append_val, args=("5",), use_container_width=True)
    r4_col3.button("6", on_click=append_val, args=("6",), use_container_width=True)
    r4_col4.button("×", on_click=append_val, args=("*",), use_container_width=True)
    r4_col5.button("x!", on_click=append_val, args=("factorial(",), use_container_width=True)
    r4_col6.button("abs", on_click=append_val, args=("abs(",), use_container_width=True)

    r5_col1, r5_col2, r5_col3, r5_col4, r5_col5, r5_col6 = st.columns(6)
    r5_col1.button("1", on_click=append_val, args=("1",), use_container_width=True)
    r5_col2.button("2", on_click=append_val, args=("2",), use_container_width=True)
    r5_col3.button("3", on_click=append_val, args=("3",), use_container_width=True)
    r5_col4.button("-", on_click=append_val, args=("-",), use_container_width=True)
    r5_col5.write("")
    r5_col6.write("")

    r6_col1, r6_col2, r6_col3, r6_col4, r6_col5, r6_col6 = st.columns(6)
    r6_col1.button("0", on_click=append_val, args=("0",), use_container_width=True)
    r6_col2.button(".", on_click=append_val, args=(".",), use_container_width=True)
    r6_col3.button("+", on_click=append_val, args=("+",), use_container_width=True)
    solve_clicked = r6_col4.button("=", type="primary", use_container_width=True)

    st.divider()

    if solve_clicked or st.button("Calculate Result", type="primary"):
        if not st.session_state.calc_input:
            st.warning("Please enter a mathematical expression.")
        else:
            try:
                allowed_scope = {
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "log": math.log10,
                    "ln": math.log,
                    "sqrt": math.sqrt,
                    "pi": math.pi,
                    "e": math.e,
                    "factorial": math.factorial,
                    "abs": abs,
                }
                
                result = eval(st.session_state.calc_input, {"__builtins__": None}, allowed_scope)
                st.subheader("Result")
                st.success(f"**{st.session_state.calc_input}** = **{result}**")
            except Exception as err:
                st.error(f"Invalid Expression: {err}")


# --- PAGE 4: TEXT HUMANIZER ---
def render_humanizer_page():
    st.title("Text Humanizer ✍️")
    st.caption("Rewrite AI-generated text into natural, varied, human sounding language.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original AI Text")
        input_text = st.text_area("Paste text here:", height=300, placeholder="Paste AI generated text...")
        
        tone = st.selectbox("Tone / Style:", ["Casual & Natural", "Academic / Formal", "Conversational Essay", "Persuasive"])
        humanize_btn = st.button("Humanize Text", type="primary", use_container_width=True)

    with col2:
        st.subheader("Humanized Output")
        if humanize_btn:
            if not input_text:
                st.warning("Please paste some text first.")
            else:
                system_prompt = (
                    "You are an expert human writer and editor. Your job is to rewrite the provided text so it sounds completely natural, "
                    "human, and authentic. Eliminate robotic phrasing, overused AI transitions (like 'in conclusion', 'delve into', 'testament to'), "
                    "and uniform sentence lengths. Mix short and long sentences, use natural phrasing, and adapt to the requested tone."
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Tone: {tone}\n\nRewrite this text to sound human:\n{input_text}"}
                ]
                
                with st.spinner("Rewriting text..."):
                    res = client.chat.completions.create(model=MODEL, messages=messages)
                    humanized_result = res.choices[0].message.content
                    st.text_area("Result:", value=humanized_result, height=300)


# --- PAGE 5: AI TEXT DETECTOR ---
def render_detector_page():
    st.title("AI Text Detector 🔍")
    st.caption("Analyze text for AI patterns, structural uniformity, and common LLM phrasing.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Text")
        detect_text = st.text_area("Paste text to analyze:", height=300, placeholder="Paste text here to evaluate if it was generated by AI...")
        analyze_btn = st.button("Analyze Text", type="primary", use_container_width=True)

    with col2:
        st.subheader("Analysis & Score")
        if analyze_btn:
            if not detect_text:
                st.warning("Please enter text to analyze.")
            else:
                system_prompt = (
                    "You are an AI text analysis model. Evaluate the user's text for markers of AI generation. "
                    "Analyze factors like burstiness, perplexity, repetitive phrasing, overused AI transition words "
                    "(e.g., 'delve', 'testament', 'crucial', 'furthermore', 'in conclusion'), uniform sentence structures, "
                    "and lack of personal nuance.\n\n"
                    "Format your response cleanly:\n"
                    "1. Estimated AI Probability (0% to 100%)\n"
                    "2. Estimated Human Probability (0% to 100%)\n"
                    "3. Highlighted AI Markers / Clues Found\n"
                    "4. Brief Final Summary"
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n\n{detect_text}"}
                ]

                with st.spinner("Analyzing text markers..."):
                    res = client.chat.completions.create(model=MODEL, messages=messages)
                    analysis_result = res.choices[0].message.content
                    st.markdown(analysis_result)


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
    st.Page(render_calculator_page, title="Scientific Calculator", icon="🧮"),
    st.Page(render_humanizer_page, title="Text Humanizer", icon="✍️"),
    st.Page(render_detector_page, title="AI Detector", icon="🔍")
])

page.run()
