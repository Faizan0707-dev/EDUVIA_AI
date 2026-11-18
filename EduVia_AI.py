import os
import re
import random
import json
import uuid
import time
import shutil
from pathlib import Path
from io import BytesIO

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import pytesseract
import fitz
from PIL import Image
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from Databasee import (
    save_user_chats, load_user_chats, delete_user_chat,
    create_user, verify_user, user_exists, change_password,
)
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
import platform

load_dotenv()

try:
    for key in ["GROQ_API_KEY", "DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


UPLOAD_ROOT = Path("uploads")
VECTORSTORE_ROOT = Path("data/vectorstores")

UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
VECTORSTORE_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="EduVia AI",
    page_icon="📚",
    layout="wide"
)



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f5f6fc;
    --white: #ffffff;
    --ink: #171a31;
    --muted: #747b98;
    --primary: #536dff;
    --primary-dark: #4055e8;
    --violet: #9674ff;
    --border: #e8eaf4;
    --soft: #f0f1ff;
    --shadow: 0 8px 30px rgba(37, 45, 100, .07);
}

/* GLOBAL */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, rgba(150,116,255,.12), transparent 35%),
        radial-gradient(ellipse at 95% 10%, rgba(83,109,255,.10), transparent 35%),
        #f5f6fc !important;
    background-attachment: fixed;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 5rem;
    animation: pageIn .65s ease both;
}

@keyframes pageIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* HEADINGS */
h1, h2, h3, h4, h5 {
    color: #191d38 !important;
    font-weight: 800 !important;
    letter-spacing: -.6px;
}

p, li {
    color: #525b78;
    line-height: 1.75;
}

small {
    color: #747b98 !important;
}

/* TOP HEADER */
header[data-testid="stHeader"] {
    background: rgba(245,246,252,.85) !important;
    backdrop-filter: blur(14px);
}

[data-testid="stDecoration"] {
    display: none;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,.96) !important;
    border-right: 1px solid #e8eaf4;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #191d38 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #414966;
}

section[data-testid="stSidebar"] button {
    border-radius: 11px !important;
    transition: all .25s ease !important;
}

section[data-testid="stSidebar"] button:hover {
    background: #eff0ff !important;
    border-color: #d9dcff !important;
    transform: translateX(3px);
}

/* BUTTONS */
.stButton > button,
.stFormSubmitButton > button,
button[kind="primary"] {
    background: linear-gradient(115deg, #536dff, #795cf5) !important;
    color: white !important;
    border: 1px solid transparent !important;
    border-radius: 11px !important;
    min-height: 43px;
    font-weight: 700 !important;
    box-shadow: 0 5px 14px rgba(83,109,255,.19);
    transition: all .25s ease !important;
}

.stButton > button *,
.stFormSubmitButton > button *,
button[kind="primary"] * {
    color: white !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 22px rgba(83,109,255,.28);
    filter: brightness(1.05);
}

.stButton > button:active {
    transform: scale(.97);
}

/* SECONDARY BUTTONS */
button[kind="secondary"] {
    background: #ffffff !important;
    color: #39415e !important;
    border: 1px solid #e2e5f1 !important;
    box-shadow: 0 3px 10px rgba(30,40,80,.04);
}

button[kind="secondary"] * {
    color: #39415e !important;
}

/* INPUTS */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stDateInput input,
.stTimeInput input {
    background: #fff !important;
    color: #191d38 !important;
    border: 1px solid #e1e4f0 !important;
    border-radius: 11px !important;
    min-height: 43px;
    transition: all .25s ease;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #7184ff !important;
    box-shadow: 0 0 0 3px rgba(83,109,255,.12) !important;
}

input::placeholder,
textarea::placeholder {
    color: #9ba2b9 !important;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: white !important;
    color: #191d38 !important;
    border: 1px solid #e1e4f0 !important;
    border-radius: 11px !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
    background: #fff !important;
    color: #191d38 !important;
}

[role="option"] {
    color: #191d38 !important;
}

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.96) !important;
    border: 1px solid #e8eaf4 !important;
    border-radius: 17px !important;
    padding: 17px !important;
    margin-bottom: 15px;
    box-shadow: 0 5px 18px rgba(30,40,90,.045);
    animation: messageIn .4s ease both;
    transition: box-shadow .25s ease, transform .25s ease;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 9px 24px rgba(50,60,130,.08);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div {
    color: #343b59;
}

@keyframes messageIn {
    from { opacity: 0; transform: translateY(9px); }
    to { opacity: 1; transform: translateY(0); }
}

/* USER CHAT BUBBLE */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(120deg,#f0f2ff,#f8f7ff) !important;
    border-color: #dfe2ff !important;
}

/* CHAT INPUT */
[data-testid="stChatInput"] {
    background: #fff !important;
    border: 1px solid #dfe3f1 !important;
    border-radius: 17px !important;
    box-shadow: 0 7px 24px rgba(40,50,100,.09);
    transition: all .25s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #7184ff !important;
    box-shadow: 0 0 0 3px rgba(83,109,255,.10),
                0 8px 25px rgba(40,50,100,.09);
}

[data-testid="stChatInput"] textarea {
    color: #191d38 !important;
    background: white !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg,#f8f8ff,#f2f4ff) !important;
    border: 1px dashed #a9b3ff !important;
    border-radius: 15px !important;
    padding: 12px;
    transition: all .25s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #536dff !important;
    background: #eff0ff !important;
}

[data-testid="stFileUploader"] * {
    color: #465071 !important;
}

[data-testid="stFileUploader"] button {
    background: #536dff !important;
    color: white !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 9px;
    border-bottom: 1px solid #e6e8f2;
}

.stTabs [data-baseweb="tab"] {
    color: #747b98 !important;
    font-weight: 700 !important;
}

.stTabs [aria-selected="true"] {
    color: #536dff !important;
    border-bottom: 3px solid #536dff !important;
}

/* CARDS */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff !important;
    border: 1px solid #e8eaf4 !important;
    border-radius: 16px !important;
    box-shadow: 0 5px 20px rgba(40,50,100,.045);
    transition: all .25s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 9px 26px rgba(40,50,100,.08);
}

/* EXPANDERS */
[data-testid="stExpander"] {
    background: #fff !important;
    border: 1px solid #e7e9f3 !important;
    border-radius: 13px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {
    color: #252b48 !important;
}

/* QUIZ OPTIONS */
.stRadio [role="radiogroup"] {
    gap: 9px;
}

.stRadio [role="radiogroup"] label {
    background: #f8f8ff !important;
    border: 1px solid #e5e7f4 !important;
    border-radius: 11px !important;
    padding: 11px 14px !important;
    transition: all .2s ease;
}

.stRadio [role="radiogroup"] label:hover {
    background: #eff0ff !important;
    border-color: #9ba7ff !important;
}

.stRadio [role="radiogroup"] label p {
    color: #343b59 !important;
}

/* CHECKBOX */
.stCheckbox label,
.stCheckbox label p {
    color: #343b59 !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background: linear-gradient(90deg,#536dff,#9674ff) !important;
    border-radius: 20px;
}

.stProgress > div {
    background: #e9eaf5 !important;
    border-radius: 20px;
}

/* ALERTS */
[data-testid="stAlert"] {
    border-radius: 13px !important;
}

[data-testid="stAlert"] p {
    color: #343b59 !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e8eaf4;
    border-radius: 15px;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(40,50,100,.04);
}

[data-testid="stMetricLabel"] {
    color: #747b98 !important;
}

[data-testid="stMetricValue"] {
    color: #242b4b !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e8f2;
    border-radius: 12px;
    overflow: hidden;
}

/* LINKS */
a {
    color: #536dff !important;
    font-weight: 600;
}

/* DIVIDERS */
hr {
    border-color: #e8eaf4 !important;
}

/* FOCUS */
button:focus-visible,
input:focus-visible {
    outline: 3px solid rgba(83,109,255,.25) !important;
    outline-offset: 2px;
}

/* MOBILE */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 5rem;
    }

    h1 {
        font-size: 1.75rem !important;
    }

    [data-testid="stChatMessage"] {
        padding: 12px !important;
    }

    .stButton > button {
        width: 100%;
    }
}

/* REDUCE MOTION ACCESSIBILITY */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        transition-duration: .01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI models...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def load_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )


@st.cache_resource(show_spinner=False)
def load_quiz_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        reasoning_effort="low",
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )


embeddings = load_embeddings()
model = load_model()
quiz_model = load_quiz_model()

cookie_password = os.getenv("COOKIE_PASSWORD")

if not cookie_password:
    try:
        cookie_password = st.secrets.get("COOKIE_PASSWORD")
    except Exception:
        cookie_password = None

if not cookie_password:
    print("CONFIG ERROR: COOKIE_PASSWORD is missing")
    st.error("Service is temporarily unavailable. Please try again later.")
    st.stop()


cookies = EncryptedCookieManager(
    prefix="eduvia_",
    password=cookie_password
)

if not cookies.ready():
    st.stop()

user_id = cookies.get("user_id")

if st.session_state.pop("logging_out", False):
    cookies["user_id"] = ""
    cookies.save()
    user_id = None

if user_id and not user_exists(user_id):
    del cookies["user_id"]
    cookies.save()
    user_id = None

if not user_id:
    st.title("📚 EduVia AI")
    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            login_id = st.text_input("Email or phone", key="login_id", autocomplete="username")
            login_pwd = st.text_input("Password", type="password", key="login_pwd", autocomplete="current-password")
            login_clicked = st.form_submit_button("Login")

        if login_clicked:
            uid = verify_user(login_id, login_pwd) if login_id and login_pwd else None
            if uid:
                st.session_state.pop("all_chats", None)
                cookies["user_id"] = uid
                cookies.save()
                st.success("Logging in. Please wait...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Incorrect email/phone or password.")

    with tab_signup:
        with st.form("Sighn up form"):
            su_name = st.text_input("Name", key="su_name", autocomplete="name")
            su_email = st.text_input("Email", key="su_email", autocomplete="email")
            su_phone = st.text_input("Phone number (10 digits)", key="su_phone", autocomplete="tel", placeholder="9876543210", max_chars=15)
            su_pwd = st.text_input("Password (6 - 72 characters )", type="password", key="su_pwd", autocomplete="new-password")
            signup_clicked = st.form_submit_button("Sign up")

        if signup_clicked:
            email = su_email.strip().lower()
            if not (su_name.strip() and su_email.strip() and su_phone.strip()):
                st.error("Please fill in all fields.")
            elif not (email.endswith("@gmail.com") and len(email) > 10 and email.count("@") == 1):
                st.error("Please enter a valid Gmail address (example@gmail.com).")
            elif len(su_pwd) < 6:
                st.error("Password must be at least 6 characters.")
            elif len(su_pwd.encode("utf-8")) > 72:
                st.error("Password is too long (max 72 bytes).")
            else:
                uid, err = create_user(su_name, su_email, su_phone, su_pwd)
                if err == "bad_email":
                    st.error("Please sign up with a valid Gmail address.")
                elif err == "bad_phone":
                    st.error("Please enter a valid 10-digit mobile number.")
                elif err == "email":
                    st.error("An account with this email already exists.")
                elif err == "phone":
                    st.error("An account with this phone number already exists.")
                elif err:
                    st.error("We couldn't create your account. Please try again.")
                else:
                    st.session_state.pop("all_chats", None)
                    cookies["user_id"] = uid
                    cookies.save()
                    st.success("Account created. Please wait...")
                    time.sleep(1)
                    st.rerun()

    st.stop()


def create_empty_chat():
    return {
        "title": "New Chat",
        "history": [],
        "quiz": [],
        "quiz_answers": {},
        "quiz_score": None,
        "quiz_submitted": False
    }


def save_chat_data():
    save_user_chats(user_id, st.session_state.all_chats)


def create_new_chat():
    chat_id = f"chat_{uuid.uuid4().hex}"

    st.session_state.all_chats[chat_id] = create_empty_chat()
    st.session_state.current_chat_id = chat_id
    st.session_state.pop("generate_quiz", None)

    save_chat_data()


def delete_chat(chat_id):
    shutil.rmtree(
        UPLOAD_ROOT / user_id / chat_id,
        ignore_errors=True
    )

    shutil.rmtree(
        VECTORSTORE_ROOT / user_id / chat_id,
        ignore_errors=True
    )

    st.session_state.all_chats.pop(chat_id, None)
    delete_user_chat(user_id,chat_id)

    if st.session_state.current_chat_id == chat_id:
        if st.session_state.all_chats:
            st.session_state.current_chat_id = next(
                iter(st.session_state.all_chats)
            )
        else:
            create_new_chat()
            return

    save_chat_data()



def extract_pdf_text(file_path):
    extracted_pages = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # OCR tabhi chalega jab extracted text insufficient ho
            if len(text) < 30:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1))

                image = Image.open(
                    BytesIO(pixmap.tobytes("png"))
                )
                image = preprocess_image(image)

                text = pytesseract.image_to_string(
                    image,
                    lang="hin+eng",
                    config="--oem 3 --psm 6"
                ).strip()

            if text:
                extracted_pages.append(
                    f"[Page {page_number}]\n{text}"
                )

    return "\n\n".join(extracted_pages)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

def preprocess_image(image, scale=2):
    from PIL import ImageEnhance, ImageFilter

    image = image.convert("L")

    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.LANCZOS
        )

    image = ImageEnhance.Contrast(image).enhance(2)
    image = image.filter(ImageFilter.SHARPEN)

    return image

def extract_image_text(image_path):
    with Image.open(image_path) as image:
        image = preprocess_image(image, scale=1)

        text = pytesseract.image_to_string(
            image,
            lang="hin+eng",
            config="--oem 3 --psm 6"
        ).strip()

    return f"[Image]\n{text}" if text else ""

def extract_file_text(file_path):
    if Path(file_path).suffix.lower() in IMAGE_EXTS:
        try:
            return extract_image_text(file_path)
        except Exception as e:
            print(f"Image OCR error: {e}")
            return ""
    return extract_pdf_text(file_path)


def ingest_files(user_id, chat_id, file_paths, upload_status, start_time):
    documents = []
    chat = st.session_state.all_chats[chat_id]
    long_message_shown = False

    def update_upload_status():
        nonlocal long_message_shown

        if not long_message_shown and time.monotonic() - start_time > 10:
            upload_status.update(
                label="Processing is taking a little longer than expected. Please wait patiently.",
                state="running"
            )
            long_message_shown = True

    for file_path in file_paths:
        text = extract_file_text(file_path)
        update_upload_status()

        if text.strip():
            file_type = (
                "Image" if Path(file_path).suffix.lower() in IMAGE_EXTS
                else "PDF"
            )

            chat["history"].append({
                "role": "system",
                "type": "file_context",
                "content": f"{file_type}: {Path(file_path).name}\n{text}"
            })

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": Path(file_path).name}
                )
            )
        else:
            st.toast(f"Could not read: {Path(file_path).name}", icon="⚠️")

    if not documents:
        st.warning("No readable text found in the uploaded files.")
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    update_upload_status()

    store_path = VECTORSTORE_ROOT / user_id / chat_id
    store_path.mkdir(parents=True, exist_ok=True)

    if (store_path / "index.faiss").exists():
        vectorstore = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    update_upload_status()

    vectorstore.save_local(str(store_path))

    save_chat_data()
    return True
    
def rag_search(query, user_id, chat_id, source=None):
    store_path = VECTORSTORE_ROOT / user_id / chat_id

    if not (store_path / "index.faiss").exists():
        return "NO_VECTORSTORE"

    vectorstore = FAISS.load_local(
        str(store_path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    q = query.lower()
    if len(q.split()) < 3 or "explain" in q or "summar" in q or "overview" in q:
        docs = [
            d for d in vectorstore.docstore._dict.values()
            if not source or d.metadata.get("source") == source
        ][:6]
        if docs:
            return "\n\n".join(d.page_content for d in docs)

    if source:
        results = vectorstore.similarity_search_with_relevance_scores(
            query, k=6, filter={"source": source}, fetch_k=200
        )
    else:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=6)

    if not results:
        return "NO_RELEVANT_CONTEXT"

    print("RAG scores:", source, [round(s, 3) for _, s in results])

    return "\n\n".join(doc.page_content for doc, score in results)

def web_search(query):

    WS = DuckDuckGoSearchRun()

    try:
        result = WS.invoke(query)
        if not result.strip():
            return "NO WEB RESULT " 
        return result
    except Exception as e:
        print(f"DuckDuckgo search error : {e}")
        return "Web Search Error"


@tool
def search_pdf(search_query: str, config: RunnableConfig) -> str:
    """Search the text extracted from PDFs and images uploaded in this chat.
        search_query must be complete and standalone (include the topic,
        never words like 'it' or 'this')."""
    cfg = config["configurable"]
    result = "NO_RELEVANT_CONTEXT"
    if cfg.get("latest_file"):
        result = rag_search(search_query, cfg["user_id"], cfg["chat_id"], cfg["latest_file"])
    if result == "NO_RELEVANT_CONTEXT":
        result = rag_search(search_query, cfg["user_id"], cfg["chat_id"])
    return result


@tool
def search_web(search_query: str) -> str:
    """Search the internet for current events, news, or when uploaded files
        have no answer, or when the student explicitly asks for the web.
        search_query must be complete and standalone."""
    return web_search(search_query)

@st.cache_resource(show_spinner=False)
def build_graph():
    model_with_tools = model.bind_tools([search_pdf, search_web])

    def call_model(state: MessagesState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([search_pdf, search_web]))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

graph = build_graph()


SYSTEM_PROMPT = """You are EduVia AI, a study assistant for students.

Tools:
- search_pdf: searches the content extracted from PDFs and images uploaded in this chat.
- search_web: searches the internet.

Rules:
1. If the student explicitly asks for the web or internet (for example "search the web", "web se batao"), call search_web directly. Do not call search_pdf.

2. If the student says to use only uploaded files or not to use the web (for example "only from uploaded files", "only from image", "only from PDF", "web se mat lana"), call only search_pdf. Never call search_web. If the uploaded files have no answer (NO_VECTORSTORE or NO_RELEVANT_CONTEXT), tell the student the answer is not in their uploaded files. If NO_VECTORSTORE, ask them to upload a file.

3. Greetings, thanks and small talk: answer directly, use no tool.

4. Every student message includes an "Uploaded file search result". If it contains content that actually answers the question, answer from it and start with "From your uploaded files:". If it is NO_VECTORSTORE, NO_RELEVANT_CONTEXT, or does not answer the question, treat it as no uploaded file answer and follow rule 5.

5. If search_pdf returns NO_VECTORSTORE or NO_RELEVANT_CONTEXT, call search_web, unless rule 2 applies.

6. Questions about current events, news, scores or recent data: call search_web directly.

7. Write every search_query as a complete, standalone sentence with the topic taken from the conversation. Never use words like 'it' or 'this'.

8. Unless rule 2 applies: if search_web returns NO WEB RESULT or Web Search Error, answer from your own general knowledge and clearly say it is from EduVia AI's own knowledge and may not be fully accurate. For live data like scores or news, say you cannot get it right now instead of answering from memory.

9. Base your answer on the tool results. Start the answer by stating the source: "From your uploaded files:", "From the web:", or "By EduVia AI:". Also add 2 lines after the source name. Never present web or own-knowledge content as if it came from uploaded files.

10. LANGUAGE RULE:
- Identify the language ONLY from the latest "Student question".
- If the latest question is in English, reply entirely in English.
- If the latest question is in Hindi, reply entirely in Hindi.
- If the latest question is in Hinglish, reply in Hinglish.
- If the student explicitly asks you to answer in a particular language (e.g., "Reply in Hindi", "Answer in English"), ALWAYS follow that language instruction, regardless of the language used in the question or previous messages.
- Ignore the language of previous conversation messages and uploaded files.
- Even if PDF or web search results are in another language, translate the information into the language required by the latest student question or explicit language instruction.
- Never randomly switch languages in the middle of an answer.

11. If the student asks to make a quiz, test or MCQs, do not call any tool. Reply: "Please click the 📝 Generate Quiz button in the sidebar." in the student's language.

12. If the student asks for something so long that it cannot fit in one reply (a very long list, essay, or many items), write a complete first part of a reasonable length and stop at a natural point. Then end with one short line in the student's language saying the full answer is very long, what you have covered so far, and that they can ask for the next part in any way they like (for example "continue", "aage ka do", "next part"). If the student asks for the next part in any language or wording, carry on exactly from where you stopped, and end with the same kind of line until everything is done.
"""


def run_agent(query, user_id, chat_id, chat_history):
    latest_file = None
    for m in chat_history:
        if m.get("type") == "file_context":
            latest_file = m["content"].split("\n", 1)[0].split(": ", 1)[1]

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    past = [
        m for m in chat_history[:-1]
        if m.get("type") != "file_context"
        and m["role"] in ("user", "assistant")
    ]
    for message in past[-10:]:
        content = message["content"][:1500]
        if message["role"] == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
            
    file_context = rag_search(query, user_id, chat_id, latest_file)

    if file_context == "NO_RELEVANT_CONTEXT" and latest_file:
        file_context = rag_search(query, user_id, chat_id)

    messages.append(HumanMessage(content=(
        f"Uploaded file search result:\n{file_context}\n\nStudent question: {query}"
)))
    try:
        result = graph.invoke(
            {"messages": messages},
            config={
                "configurable": {
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "latest_file": latest_file
                },
                "recursion_limit": 10
            }
        )
        return result["messages"][-1].content
    except GraphRecursionError:
        return "Sorry, I could not finish this request. Please rephrase your question."
    except Exception as e:
        import traceback
        traceback.print_exc()
        err = str(e)
        if getattr(e, "status_code", None) == 413 or "too large" in err.lower():
            return "Your message is too long. Please send a shorter message."
        if getattr(e, "status_code", None) == 429:
            wait = re.search(r"try again in ([0-9dhms.]+)", err)
            if wait:
                return f"Maximum limit reached. Please try again in {wait.group(1).rstrip('.')}."
            return "Maximum limit reached. Please try again after some time."
        return "Some problem occured please try again"
    
def generate_quiz(user_id, chat_id, chat_history):

    chat = st.session_state.all_chats.get(chat_id)

    if chat is None:
        st.error("Chat not found.")
        return

    chat.setdefault("quiz_question_history", [])

    if not st.session_state.get("generate_quiz", False):
        return

    recent_messages = [
        message for message in chat_history
        if message.get("role") in ("user", "assistant")
        and message.get("type") != "file_context"
    ]

    conversation_chunks = []

    for message in recent_messages[-4:]:
        content = message.get("content", "").strip()

        if len(content) >= 100:
            conversation_chunks.append(content[-700:])

    chat_text = "\n\n".join(conversation_chunks)

    file_batches = []
    current_batch = []

    for message in chat_history:
        if message.get("type") == "file_context":
            current_batch.append(message)
        elif current_batch:
            file_batches.append(current_batch)
            current_batch = []

    if current_batch:
        file_batches.append(current_batch)

    latest_file_batch = file_batches[-1] if file_batches else []
    older_file_batches = file_batches[:-1]

    latest_file_chunks = []

    for message in latest_file_batch:
        content = message.get("content", "").strip()

        for start in range(0, len(content), 1000):
            chunk = content[start:start + 1000].strip()

            if len(chunk) >= 100:
                latest_file_chunks.append(chunk)

            if len(latest_file_chunks) >= 4:
                break

        if len(latest_file_chunks) >= 4:
            break

    latest_file_text = "\n\n".join(latest_file_chunks)

    older_file_chunks = []

    for batch in reversed(older_file_batches):
        for message in batch:
            content = message.get("content", "").strip()

            for start in range(0, len(content), 800):
                chunk = content[start:start + 800].strip()

                if len(chunk) >= 100:
                    older_file_chunks.append(chunk)

                if len(older_file_chunks) >= 2:
                    break

            if len(older_file_chunks) >= 2:
                break

        if len(older_file_chunks) >= 2:
            break

    older_file_text = "\n\n".join(older_file_chunks)

    context = (chat_text + "\n\n" + latest_file_text).strip()

    if len(context) < 800 and older_file_text:
        context = (context + "\n\n" + older_file_text).strip()
    else:
        older_file_text = ""
    if len(context) < 800:
        st.warning(
            "Not enough study material is available in this chat. "
            "Please upload a PDF or image, or continue the conversation."
        )
        return
    quiz_source = context
    
    previous_questions = []


    for old_quiz in chat["history"]:
        if old_quiz.get("type") == "quiz":
            for question in old_quiz.get("questions", []):
                question_text = question.get("question", "").strip()

                if question_text and question_text not in previous_questions:
                    previous_questions.append(question_text)

    previous_questions = previous_questions[-15:]
    prompt = f"""
You are an educational quiz generator.

Use only the study material provided below.

Generate 5 to 10 fresh multiple-choice questions.

Rules:
- Never repeat previous questions.
- Use both recent conversation and the latest uploaded file material when both are available.
- Give priority to the latest conversation while also giving substantial importance to the latest uploaded PDF or image.
- Aim for approximately 60% questions from recent conversation and 40% from the latest uploaded file material.
- Never generate the entire quiz only from uploaded files when recent conversation is available.
- Use older uploaded files only as fallback material when the recent conversation and latest uploaded file do not provide enough relevant content for a complete quiz. Prefer the recent conversation first and the latest uploaded file second. Do not use older files when the recent conversation and latest uploaded file already provide sufficient material.
- If one source has insufficient content, use more content from the other available source.
- If only one source is available, generate questions from that source.
- Do not test the same fact by changing its wording.
- Use only facts supported by the study material.
- Do not invent facts or use outside knowledge.
- Each question must have exactly 4 different options.
- Each question must have exactly one correct answer.
- The answer must exactly match one of the four options.
- If fewer than 5 distinct questions are possible, return insufficient context.
- Write everything in English.
- Return valid JSON only.

Required JSON format:

{{
  "enough_context": true,
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Option A"
    }}
  ]
}}

Previous questions:
{json.dumps(previous_questions, ensure_ascii=False)}

Recent conversation:
{chat_text}

Latest uploaded file material (PDF or image):
{latest_file_text}

Older uploaded file material (fallback only):
{older_file_text}
"""

    try:
        try:
            response = quiz_model.invoke(prompt)

        except Exception as retry_error:
            retry_err = str(retry_error).lower()

            if (
                "tpm" in retry_err
                or "tokens per minute" in retry_err
                or "request too large" in retry_err
                or "413" in retry_err
            ):
                compact_prompt = prompt.replace(
                    f"Recent conversation:\n{chat_text}",
                    f"Recent conversation:\n{chat_text[-1000:]}"
                )

                compact_prompt = compact_prompt.replace(
                    f"Latest uploaded file material (PDF or image):\n{latest_file_text}",
                    f"Latest uploaded file material (PDF or image):\n{latest_file_text[:1500]}"
                )

                compact_prompt = compact_prompt.replace(
                    f"Older uploaded file material (fallback only):\n{older_file_text}",
                    "Older uploaded file material (fallback only):\n"
                )

                response = quiz_model.invoke(compact_prompt)

            else:
                raise

        raw = response.content

        if isinstance(raw, list):
            raw = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw
            )

        raw = str(raw).strip()

        print("QUIZ RAW RESPONSE:", repr(raw))

        if not raw:
            raise ValueError("Groq returned an empty response. Try again.")

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(f"Groq did not return valid JSON: {raw[:300]}")

        raw = raw[start:end + 1]

        result = json.loads(raw)

        if not isinstance(result, dict):
            raise ValueError("Invalid JSON response.")

        if not result.get("enough_context", False):
            st.warning("Not enough material for 5 distinct questions.")
            return

        questions = result.get("questions", [])

        if not isinstance(questions, list) or not 5 <= len(questions) <= 10:
            raise ValueError("Quiz must contain 5 to 10 questions.")

        for question in questions:

            if not isinstance(question, dict):
                raise ValueError("Invalid question format.")

            options = question.get("options", [])
            answer = question.get("answer", "")
            question_text = question.get("question", "")

            if (
                not isinstance(question_text, str)
                or not question_text.strip()
                or not isinstance(options, list)
                or len(options) != 4
                or any(
                    not isinstance(option, str) or not option.strip()
                    for option in options
                )
                or len(set(options)) != 4
                or not isinstance(answer, str)
                or answer not in options
            ):
                raise ValueError("Invalid question or answer format.")

        previous_normalized = {
            " ".join(q.lower().split())
            for q in previous_questions
        }

        seen = set(previous_normalized)
        fresh_questions = []

        for q in questions:
            key = " ".join(q["question"].lower().split())

            if key in seen:
                continue

            seen.add(key)
            fresh_questions.append(q)

        if len(fresh_questions) < 5:
            st.warning("Repeated questions detected. Generate again.")
            return

        questions = fresh_questions

        quiz_id = uuid.uuid4().hex

        chat["history"].append({
            "role": "quiz",
            "type": "quiz",
            "content": quiz_id,
            "quiz_id": quiz_id,
            "questions": questions,
            "quiz_source": quiz_source,
            "answers": {},
            "score": None,
            "submitted": False
        })

        chat["quiz_question_history"] = previous_questions + [
            q["question"] for q in questions
        ]

        save_chat_data()
        st.toast("Quiz generated successfully!", icon="✅")
        st.rerun()

    except Exception as error:
        err = str(error).lower()
        print(f"Quiz generation error: {error}")

        if (
            "tpm" in err
            or "tokens per minute" in err
            or "request too large" in err
            or "413" in err
        ):
            st.error(
                "Quiz generation is temporarily limited. "
                "Please try again shortly."
            )

        elif "429" in err or "rate_limit" in err:
            st.error(
                "Quiz generation is temporarily limited. "
                "Please try again in a few moments."
            )

        else:
            st.error(
                "Quiz generation failed due to a temporary issue. "
                "Please try again."
            )

    finally:
        st.session_state.generate_quiz = False

def render_quiz(quiz, chat_id, idx=0):

    questions = quiz["questions"]
    quiz_id = quiz["quiz_id"]
    uid = f"{chat_id}_{quiz_id}_{idx}"

    with st.expander("📝 Study Quiz", expanded=True):

        with st.form(f"quiz_{uid}"):

            answers = {}

            for i, q in enumerate(questions):
                saved = quiz["answers"].get(str(i))

                answers[i] = st.radio(
                    f"Q{i+1}. {q['question']}",
                    q["options"],
                    index=q["options"].index(saved)
                    if saved in q["options"] else None,
                    key=f"{uid}_{i}",
                    disabled=quiz["submitted"]
                )

                if quiz["submitted"]:
                    selected = quiz["answers"].get(str(i))
                    correct = q["answer"]

                    if selected == correct:
                        st.success(f"Correct: {correct}")
                    else:
                        st.error(f"Your answer: {selected}")
                        st.success(f"Correct answer: {correct}")

            submitted = st.form_submit_button(
                "Submit Quiz",
                disabled=quiz["submitted"],
                key=f"submit_{uid}"
            )

        if submitted:
            if any(a is None for a in answers.values()):
                st.warning("Please answer all questions.")
                return

            quiz["answers"] = {
                str(i): a for i, a in answers.items()
            }

            quiz["score"] = sum(
                answers[i] == q["answer"]
                for i, q in enumerate(questions)
            )

            quiz["submitted"] = True

            save_chat_data()
            st.rerun()

        if quiz["submitted"]:
            score = quiz["score"]
            total = len(questions)

            st.success(f"🏆 Your Score: {score} / {total}")

            percentage = (score / total) * 100

            if percentage >= 80:
                st.success("🌟 Excellent! Outstanding performance.")
            elif percentage >= 60:
                st.success("👏 Very Good! Keep it up.")
            elif percentage >= 40:
                st.info("👍 Good effort! Keep practicing.")
            else:
                st.warning("📚 Keep learning. You can improve!")
if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_user_chats(user_id)

if not isinstance(st.session_state.all_chats, dict):
    st.session_state.all_chats = {}

for chat in st.session_state.all_chats.values():
    chat.setdefault("title", "New Chat")
    chat.setdefault("history", [])
    chat.setdefault("quiz", [])
    chat.setdefault("quiz_answers", {})
    chat.setdefault("quiz_score", None)
    chat.setdefault("quiz_submitted", False)
    chat.setdefault("quiz_question_history", [])
if (
    "current_chat_id" not in st.session_state
    or st.session_state.current_chat_id not in st.session_state.all_chats
):
    if st.session_state.all_chats:
        st.session_state.current_chat_id = next(
            iter(st.session_state.all_chats)
        )
    else:
        create_new_chat()

current_chat_id = st.session_state.current_chat_id
current_chat = st.session_state.all_chats[current_chat_id]


with st.sidebar:
    st.title("📚 EduVia AI")

    if st.button("➕ New Chat", key="new_chat" ,use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()
    st.subheader("Chat History")

    for chat_id, chat in list(st.session_state.all_chats.items()):
        title = chat.get("title", "New Chat")

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                title,
                key=f"open_{chat_id}",
                use_container_width=True
            ):
                st.session_state.current_chat_id = chat_id
                st.session_state.pop("generate_quiz", None)
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

    st.divider()

    if st.button("📝 Generate Quiz", key="gen_quiz", use_container_width=True):
        st.session_state.generate_quiz = True
        st.rerun()
    st.divider()

    if st.button("🚪 Log Out", key="logout", use_container_width=True):
        for key in ("all_chats", "current_chat_id", "generate_quiz"):
            st.session_state.pop(key, None)
        st.session_state["logging_out"] = True
        st.success("Logging out Please wait...")
        time.sleep(1)
        st.rerun()

    st.divider()

    with st.expander("🔑 Change password"):
        with st.form("change pwd form ", clear_on_submit= True):
            old_pwd = st.text_input("Old password" , type = "password" , autocomplete= "old password")
            new_pwd = st.text_input("new password (min 6 characters)" , type = "password" , autocomplete= "new password")
            con_pwd = st.text_input("Confirm password" , type = "password" , autocomplete= "new password")
            change_clicked = st.form_submit_button("Update password")
        if change_clicked:
            if not (old_pwd and new_pwd and con_pwd):
                st.error("Please fill in all fields")
            elif len(new_pwd) < 6:
                st.error("Password must be at least 6 characters.")
            elif len(new_pwd.encode("utf-8"))> 72:
                st.error("Password is too long (max 72 bytes).")
            elif new_pwd != con_pwd:
                st.error("New passwords do not match.")
            elif new_pwd == old_pwd:
                st.error("New password must be different.")
            elif change_password(user_id, old_pwd, new_pwd) == "wrong_old":
                st.error("Current password is incorrect.")
            else:
                    st.success("Password updated.")
st.title(current_chat.get("title", "New Chat"))

quiz_index = 0

for message in current_chat.get("history", []):

    if message.get("type") == "file_context":
        continue

    if message.get("type") == "quiz":
        render_quiz(message, current_chat_id, quiz_index)
        quiz_index += 1
        continue

    if message.get("role") in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.get("generate_quiz", False):
    with st.spinner("Quiz is generating, please wait..."):
        generate_quiz(
            user_id,
            current_chat_id,
            current_chat["history"]
        )

    st.session_state.generate_quiz = False
user_input = st.chat_input(
    "Ask EduVia AI...",
    accept_file="multiple",
    file_type=["pdf", "png", "jpg", "jpeg", "webp"]
)

if user_input:
    if hasattr(user_input, "text"):
        prompt = user_input.text or ""
        uploaded_files = user_input.files
    else:
        prompt = str(user_input)
        uploaded_files = []

    file_paths = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith((".pdf",) + IMAGE_EXTS):
            continue

        upload_dir = UPLOAD_ROOT / user_id / current_chat_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(uploaded_file.name).name
        saved_path = upload_dir / safe_name

        with open(saved_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        file_paths.append(str(saved_path))

    if file_paths:
        start_time = time.monotonic()

        with st.status(
            "Uploading and processing your file... Please wait.",
            expanded=False
        ) as upload_status:

            uploaded = ingest_files(
                user_id,
                current_chat_id,
                file_paths,
                upload_status,
                start_time
            )

            if uploaded:
                upload_status.update(
                    label="File processing finished!",
                    state="complete"
                )
            else:
                upload_status.update(
                    label="No readable text found.",
                    state="error"
                )

        if uploaded:
            st.success("Uploaded file content saved successfully.")
    if prompt.strip():
        current_chat["history"].append({
            "role": "user",
            "content": prompt
        })

        if current_chat.get("title") == "New Chat":
            current_chat["title"] = prompt.strip()[:40]

        save_chat_data()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("EduVia AI is thinking..."):
                response = run_agent(
                    prompt,
                    user_id,
                    current_chat_id,
                    current_chat["history"]
                )

            placeholder = st.empty()
            shown = ""
            lines = response.split("\n")
            delay = min(0.15, 3 / max(len(lines), 1))
            for line in lines:
                shown += line + "  \n"
                placeholder.markdown(shown)
                time.sleep(delay)

        current_chat["history"].append({
            "role": "assistant",
            "content": response
        })

    elif file_paths:
        current_chat["history"].append({
            "role": "user",
            "content": f"Uploaded {len(file_paths)} file(s)."
        })

    save_chat_data()
    st.rerun()
