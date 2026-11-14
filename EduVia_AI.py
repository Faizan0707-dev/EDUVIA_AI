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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

:root{
  --bg-0:#030805; --bg-1:#07130b;
  --neon:#39ff14; --neon2:#00ff9d; --neon3:#00e5ff;
  --ink:#021208; --user-a:#066b2c; --user-b:#0a8a3c;
  --ai-bg:rgba(10,28,18,.78); --line:rgba(57,255,20,.22);
  --red-a:#ff3b57; --red-b:#c4122f;
}

/* Base */
html,body,[class*="css"],.stApp{
  font-family:'Plus Jakarta Sans',sans-serif;
  color:#fff;
}
.stApp{
  background:
    radial-gradient(900px 520px at 8% -10%,rgba(57,255,20,.16),transparent 60%),
    radial-gradient(800px 520px at 100% 0%,rgba(0,229,255,.10),transparent 55%),
    radial-gradient(700px 600px at 50% 125%,rgba(0,255,157,.10),transparent 60%),
    linear-gradient(180deg,var(--bg-1),var(--bg-0));
  background-attachment:fixed;
}
header[data-testid="stHeader"]{background:transparent;z-index:999990}
[data-testid="stDecoration"]{display:none}
.block-container{padding-top:2rem;padding-bottom:7rem;max-width:1000px}
h1,h2,h3,h4{
  font-family:'Space Grotesk',sans-serif;
  letter-spacing:-.02em;color:#fff;
}
p,li,label,small,
div[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"]{color:#fff}
::selection{background:var(--neon);color:#000}
.block-container h1{
  font-weight:700;
  background:linear-gradient(90deg,#fff,#b6ffa8 45%,#39ff14);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;
}

/* Toolbar: Rerun, Deploy and menu */
[data-testid="stToolbar"] *,
[data-testid="stStatusWidget"] *{
  color:#fff!important;
  fill:#fff!important;
  -webkit-text-fill-color:#fff!important;
}
[data-testid="stToolbar"] button,
[data-testid="stStatusWidget"] button{
  color:#fff!important;
  background:rgba(57,255,20,.10)!important;
  border:1px solid var(--line)!important;
}
[data-testid="stToolbar"] button p,
[data-testid="stStatusWidget"] button p{
  color:#fff!important;
  -webkit-text-fill-color:#fff!important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#07170d 0%,#030a05 100%);
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] h1{
  font-family:'Space Grotesk',sans-serif;
  font-size:1.9rem;font-weight:700;
  background:linear-gradient(90deg,#fff,#8dff6b 45%,#39ff14);
  -webkit-background-clip:text;background-clip:text;
  -webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 14px rgba(57,255,20,.55));
}
section[data-testid="stSidebar"] h3{
  font-size:.95rem;font-weight:600;color:rgba(255,255,255,.75);
}
section[data-testid="stSidebar"] hr{border-color:var(--line);margin:.9rem 0}

/* General buttons */
.stApp .stButton>button,
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid^="stBaseButton"]{
  border-radius:14px;
  border:1px solid rgba(57,255,20,.45);
  background:rgba(57,255,20,.10);
  color:#fff;font-weight:600;
  transition:transform .18s ease,box-shadow .18s ease,background .18s ease;
}
.stApp .stButton>button p,
.stApp [data-testid="stFormSubmitButton"] button p{color:inherit}
.stApp .stButton>button:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover{
  border-color:var(--neon);
  box-shadow:0 6px 22px rgba(57,255,20,.35);
  transform:translateY(-1px);
}
.stApp button:disabled{
  opacity:.65!important;color:#fff!important;
  background:rgba(57,255,20,.12)!important;
  border:1px solid rgba(57,255,20,.35)!important;
  box-shadow:none!important;transform:none!important;
  cursor:not-allowed;
}
.stApp button:disabled p{color:#fff!important}

/* Form submit buttons */
.stApp [data-testid="stFormSubmitButton"] button:not(:disabled){
  background:linear-gradient(135deg,#39ff14,#00ff9d)!important;
  border:none!important;color:var(--ink)!important;
  font-weight:700;box-shadow:0 6px 20px rgba(57,255,20,.40);
}
.stApp [data-testid="stFormSubmitButton"] button:not(:disabled) p{
  color:var(--ink)!important;
}

/* Chat history buttons */
.stApp [class*="st-key-open_"] button{
  justify-content:flex-start;text-align:left;
  padding:.65rem .9rem;
  background:linear-gradient(135deg,rgba(57,255,20,.14),rgba(0,229,255,.08))!important;
  border:1px solid rgba(57,255,20,.32)!important;color:#fff!important;
}
.stApp [class*="st-key-open_"] button p{
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  width:100%;color:#fff!important;
}
.stApp [class*="st-key-open_"] button:hover{
  transform:translateX(4px);
  border-color:var(--neon)!important;
  background:linear-gradient(135deg,rgba(57,255,20,.30),rgba(0,229,255,.16))!important;
  box-shadow:0 6px 22px rgba(57,255,20,.30);
}

/* New Chat */
.stApp .st-key-new_chat button{
  border:none!important;
  background:linear-gradient(120deg,#39ff14,#00ff9d,#00e5ff,#39ff14)!important;
  background-size:300% 300%!important;
  animation:eduShift 6s ease infinite;
  color:var(--ink)!important;font-weight:800!important;
  letter-spacing:.2px;box-shadow:0 8px 26px rgba(57,255,20,.45);
}
.stApp .st-key-new_chat button p{color:var(--ink)!important;font-weight:800}
.stApp .st-key-new_chat button:hover{
  transform:translateY(-2px) scale(1.02);
  box-shadow:0 12px 34px rgba(57,255,20,.65);
}
@keyframes eduShift{
  0%{background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}

/* Generate Quiz */
.stApp .st-key-gen_quiz button{
  border:1px solid rgba(0,229,255,.7)!important;
  background:linear-gradient(135deg,rgba(0,229,255,.22),rgba(57,255,20,.14))!important;
  color:#fff!important;
}
.stApp .st-key-gen_quiz button p{color:#fff!important}
.stApp .st-key-gen_quiz button:hover{
  transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(0,229,255,.40);
}

/* Delete and Logout */
.stApp [class*="st-key-delete_"] button,
.stApp .st-key-logout button{
  background:linear-gradient(135deg,var(--red-a),var(--red-b))!important;
  border:none!important;color:#fff!important;
  box-shadow:0 6px 18px rgba(255,59,87,.35);
}
.stApp [class*="st-key-delete_"] button p,
.stApp .st-key-logout button p{
  color:#fff!important;font-weight:600;
}
.stApp [class*="st-key-delete_"] button:hover,
.stApp .st-key-logout button:hover{
  transform:translateY(-2px);
  box-shadow:0 10px 26px rgba(255,59,87,.55);
  filter:brightness(1.1);
}

/* Chat messages */
[data-testid="stChatMessage"]{
  background:transparent;padding:.5rem .25rem;
  gap:.8rem;align-items:flex-start;width:100%;
}
[data-testid="stChatMessageContent"]{
  padding:.85rem 1.1rem;border-radius:20px;
  line-height:1.65;font-size:1rem;color:#fff;
  box-shadow:0 8px 28px rgba(0,0,0,.35);min-width:0;
}
[data-testid="stChatMessageContent"] *{color:#fff}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  flex-direction:row!important;justify-content:flex-start!important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]{
  flex:1 1 0!important;width:auto!important;max-width:100%;
  background:var(--ai-bg);border:1px solid var(--line);
  border-top-left-radius:6px;backdrop-filter:blur(10px);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  flex-direction:row-reverse!important;
  justify-content:flex-start!important;text-align:left;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  flex:0 1 auto!important;width:auto!important;max-width:75%;
  margin:0!important;
  background:linear-gradient(135deg,var(--user-a),var(--user-b));
  border:1px solid rgba(57,255,20,.55);
  border-top-right-radius:6px;
  box-shadow:0 0 22px rgba(57,255,20,.25),0 10px 30px rgba(0,0,0,.35);
}

/* Chat avatars */
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"]{
  width:46px;height:46px;min-width:46px;
  flex-shrink:0;border-radius:50%;position:relative;
  display:flex;align-items:center;justify-content:center;
  overflow:visible;border:none;transition:transform .25s ease;
}
[data-testid="stChatMessageAvatarAssistant"]>*,
[data-testid="stChatMessageAvatarUser"]>*{display:none!important}
[data-testid="stChatMessageAvatarAssistant"]::before,
[data-testid="stChatMessageAvatarUser"]::before{
  font-size:27px;line-height:1;width:46px;height:46px;
  border-radius:50%;display:flex;align-items:center;justify-content:center;
  filter:drop-shadow(0 3px 3px rgba(0,0,0,.45));
}
[data-testid="stChatMessageAvatarAssistant"]::before{
  content:"🤖";
  background:
    radial-gradient(circle at 30% 25%,rgba(255,255,255,.75),transparent 38%),
    radial-gradient(circle at 70% 80%,#047857,transparent 60%),
    linear-gradient(145deg,#39ff14,#00b86b);
  box-shadow:inset -5px -7px 12px rgba(0,40,10,.55),
    inset 4px 4px 8px rgba(255,255,255,.35),
    0 8px 20px rgba(57,255,20,.55);
}
[data-testid="stChatMessageAvatarUser"]::before{
  content:"🧑‍💻";
  background:
    radial-gradient(circle at 30% 25%,rgba(255,255,255,.75),transparent 38%),
    radial-gradient(circle at 70% 80%,#0e7490,transparent 60%),
    linear-gradient(145deg,#00e5ff,#0284c7);
  box-shadow:inset -5px -7px 12px rgba(0,20,50,.55),
    inset 4px 4px 8px rgba(255,255,255,.35),
    0 8px 20px rgba(0,229,255,.5);
}
[data-testid="stChatMessage"]:hover [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"]:hover [data-testid="stChatMessageAvatarUser"]{
  transform:translateY(-3px) rotate(-6deg);
}
[data-testid="stChatMessageContent"] code{
  background:rgba(0,0,0,.45);color:#8dff6b;
  padding:.1rem .4rem;border-radius:6px;
}
[data-testid="stChatMessageContent"] pre{
  background:rgba(0,0,0,.55);
  border:1px solid var(--line);border-radius:12px;
}
[data-testid="stChatMessageContent"] a{color:#00e5ff!important}

/* Chat input */
[data-testid="stBottom"],
[data-testid="stBottom"]>div{background:transparent!important}
[data-testid="stBottom"]::before{
  content:"";position:absolute;inset:-30px 0 0;
  pointer-events:none;
  background:linear-gradient(180deg,transparent,var(--bg-0) 70%);
}
[data-testid="stChatInput"]{
  position:relative;
  background:linear-gradient(#07170d,#07170d) padding-box,
    linear-gradient(120deg,#39ff14,#00ff9d,#00e5ff) border-box;
  border:2px solid transparent;border-radius:28px;
  padding:.2rem .35rem;
  box-shadow:0 12px 40px rgba(57,255,20,.22),0 0 0 6px rgba(57,255,20,.06);
  transition:box-shadow .25s ease;
}
[data-testid="stChatInput"]:focus-within{
  box-shadow:0 14px 48px rgba(57,255,20,.50),0 0 0 7px rgba(57,255,20,.12);
}
[data-testid="stChatInput"]>div{background:transparent!important;border:none!important}
[data-testid="stChatInput"] textarea{
  color:#fff!important;background:transparent!important;
  font-size:1rem;caret-color:var(--neon);
}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(255,255,255,.55)}
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button[kind="primary"]{
  width:42px;height:42px;border-radius:50%!important;
  border:none!important;
  background:linear-gradient(135deg,#39ff14,#00ff9d)!important;
  color:var(--ink)!important;
  box-shadow:0 6px 18px rgba(57,255,20,.55);
  transition:transform .2s ease,box-shadow .2s ease;
}
[data-testid="stChatInputSubmitButton"] svg,
[data-testid="stChatInput"] button[kind="primary"] svg{
  fill:var(--ink);color:var(--ink);
}
[data-testid="stChatInputSubmitButton"]:hover,
[data-testid="stChatInput"] button[kind="primary"]:hover{
  transform:scale(1.1) rotate(-8deg);
  box-shadow:0 8px 26px rgba(57,255,20,.8);
}
[data-testid="stChatInputSubmitButton"]:disabled{opacity:.5!important;box-shadow:none}
[data-testid="stChatInput"] button:not([kind="primary"]):not([data-testid="stChatInputSubmitButton"]),
[data-testid="stChatInputFileUploadButton"]{
  width:40px;height:40px;border-radius:50%!important;
  background:rgba(57,255,20,.12)!important;
  border:1px solid rgba(57,255,20,.45)!important;
  color:#fff!important;transition:all .2s ease;
}
[data-testid="stChatInput"] button:not([kind="primary"]):not([data-testid="stChatInputSubmitButton"]) svg,
[data-testid="stChatInputFileUploadButton"] svg{fill:#fff;color:#fff}
[data-testid="stChatInput"] button:not([kind="primary"]):not([data-testid="stChatInputSubmitButton"]):hover,
[data-testid="stChatInputFileUploadButton"]:hover{
  background:rgba(57,255,20,.28)!important;
  border-color:var(--neon)!important;
  transform:rotate(90deg) scale(1.08);
}
[data-testid="stChatInput"] [data-testid="stChatInputFile"],
[data-testid="stChatInput"] [data-testid="stFileChip"]{
  background:rgba(57,255,20,.16);
  border:1px solid rgba(57,255,20,.5);
  color:#fff;border-radius:12px;
}

/* Expander */
[data-testid="stExpander"]{
  background:var(--ai-bg);border:1px solid var(--line);
  border-radius:18px;backdrop-filter:blur(10px);
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary{
  background:transparent!important;color:#fff!important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span{
  color:#fff!important;font-weight:600;
}
[data-testid="stExpander"] summary svg{fill:var(--neon);color:var(--neon)}
[data-testid="stExpander"] summary:hover{background:rgba(57,255,20,.08)!important}

/* Forms and inputs */
div[data-testid="stForm"]{
  background:rgba(57,255,20,.04);
  border:1px solid var(--line);border-radius:18px;
}
[data-baseweb="input"],[data-baseweb="base-input"]{
  background:rgba(255,255,255,.07)!important;
  border-radius:12px!important;border-color:var(--line)!important;
}
[data-baseweb="input"] input{
  color:#fff!important;-webkit-text-fill-color:#fff!important;
}
[data-baseweb="input"] input::placeholder{
  color:rgba(255,255,255,.5)!important;
  -webkit-text-fill-color:rgba(255,255,255,.5)!important;
}
[data-baseweb="input"]:focus-within{
  border-color:var(--neon)!important;
  box-shadow:0 0 0 2px rgba(57,255,20,.25);
}
[data-baseweb="input"] button{
  background:transparent!important;border:none!important;box-shadow:none!important;
}
[data-baseweb="input"] button svg{fill:#fff!important;color:#fff!important}

/* Radio buttons */
div[role="radiogroup"] label{
  color:#fff;padding:.3rem .6rem;border-radius:10px;
  transition:background .15s ease;
}
div[role="radiogroup"] label:hover{background:rgba(57,255,20,.10)}
div[role="radiogroup"] label p{color:#fff!important}
div[role="radiogroup"] label>div:first-child{border-color:var(--neon)!important}
div[role="radiogroup"] label>div:first-child>div{background-color:var(--neon)!important}

/* Login / Signup tabs */
.stTabs [data-baseweb="tab"]{color:rgba(255,255,255,.7);font-weight:600}
.stTabs [data-baseweb="tab"] p{color:inherit}
.stTabs [aria-selected="true"]{color:#fff}
.stTabs [data-baseweb="tab-highlight"]{
  background:linear-gradient(90deg,#39ff14,#00ff9d);height:3px;
}
.stTabs [data-baseweb="tab-border"]{background:var(--line)}

/* Alerts */
[data-testid="stAlert"]{
  background:#08160e!important;border:1px solid var(--line)!important;
  border-radius:14px!important;
  box-shadow:0 6px 20px rgba(0,0,0,.35);
}
[data-testid="stAlert"]>div,
[data-testid="stAlertContainer"]{background:transparent!important;color:#fff!important}
[data-testid="stAlert"] *{color:#fff!important}
[data-testid="stAlert"] svg{fill:currentColor}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){
  background:linear-gradient(rgba(57,255,20,.14),rgba(57,255,20,.14)),#08160e!important;
  border-color:rgba(57,255,20,.65)!important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){
  background:linear-gradient(rgba(255,59,87,.18),rgba(255,59,87,.18)),#12080b!important;
  border-color:rgba(255,59,87,.75)!important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){
  background:linear-gradient(rgba(255,196,0,.16),rgba(255,196,0,.16)),#151004!important;
  border-color:rgba(255,196,0,.75)!important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){
  background:linear-gradient(rgba(0,229,255,.14),rgba(0,229,255,.14)),#06131a!important;
  border-color:rgba(0,229,255,.7)!important;
}

/* Toast */
[data-testid="stToast"],
[data-testid="stToastContainer"]>div{
  background:#08160e!important;
  border:1px solid rgba(57,255,20,.6)!important;
  border-radius:14px!important;
  box-shadow:0 10px 30px rgba(0,0,0,.5),0 0 18px rgba(57,255,20,.25);
}
[data-testid="stToast"] *{color:#fff!important}

/* Loading spinner / Loading AI models */
[data-testid="stSpinner"],.stSpinner{
  color:#fff!important;
}
[data-testid="stSpinner"]{
  z-index:999999!important;
  background:#08160e!important;
  border:1px solid rgba(57,255,20,.6);
  border-radius:16px;padding:20px 30px!important;
  box-shadow:0 10px 35px rgba(0,0,0,.6),0 0 22px rgba(57,255,20,.25);
  max-width:90vw;
}
[data-testid="stSpinner"] *,
.stSpinner *{color:#fff!important}
[data-testid="stSpinner"] p,
.stSpinner p{
  color:#fff!important;font-size:1.25rem!important;
  font-weight:700!important;text-align:center!important;
  -webkit-text-fill-color:#fff!important;
}
[data-testid="stSpinner"] svg,
.stSpinner svg{color:var(--neon)!important}
[data-testid="stSpinner"] i,
.stSpinner i{border-top-color:var(--neon)!important}

/* Scrollbar */
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{
  background:linear-gradient(#39ff14,#00ff9d);border-radius:8px;
}

/* Sidebar open / close icons */
span[data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] span{
  font-family:'Material Symbols Rounded','Material Icons'!important;
  font-weight:400!important;letter-spacing:normal!important;
}
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] button{
  display:flex!important;visibility:visible!important;opacity:1!important;
  align-items:center;justify-content:center;
  width:42px;height:42px;margin:.5rem;
  border-radius:14px!important;border:none!important;
  background:linear-gradient(135deg,#39ff14,#00ff9d)!important;
  box-shadow:0 6px 20px rgba(57,255,20,.5);
  transition:transform .2s ease,box-shadow .2s ease;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button{
  opacity:1!important;visibility:visible!important;
}
[data-testid="stSidebarCollapseButton"] button{
  width:38px;height:38px;border-radius:12px!important;
  border:none!important;
  background:linear-gradient(135deg,#39ff14,#00ff9d)!important;
  box-shadow:0 4px 14px rgba(57,255,20,.45);
}
[data-testid="stExpandSidebarButton"] *,
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] *{
  color:var(--ink)!important;fill:var(--ink)!important;
  -webkit-text-fill-color:var(--ink)!important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover{
  transform:scale(1.1);box-shadow:0 8px 26px rgba(57,255,20,.75);
}

/* Mobile */
@media(max-width:640px){
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{max-width:85%}
  [data-testid="stChatMessageAvatarAssistant"],
  [data-testid="stChatMessageAvatarUser"]{width:38px;height:38px;min-width:38px}
  [data-testid="stChatMessageAvatarAssistant"]::before,
  [data-testid="stChatMessageAvatarUser"]::before{width:38px;height:38px;font-size:22px}
  [data-testid="stSpinner"]{padding:16px 20px!important}
  [data-testid="stSpinner"] p{font-size:1.1rem!important}
}
@media(prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
}

/* Rerun and Deploy toolbar controls */
header[data-testid="stHeader"] [data-testid="stToolbar"] button,
header[data-testid="stHeader"] [data-testid="stStatusWidget"] button,
header[data-testid="stHeader"] [data-testid="stAppDeployButton"] button{
    background:#07130b!important;
    background-color:#07130b!important;
    border:1px solid #39ff14!important;
    color:#fff!important;
    box-shadow:0 2px 10px rgba(57,255,20,.2)!important;
}

header[data-testid="stHeader"] [data-testid="stToolbar"] button svg,
header[data-testid="stHeader"] [data-testid="stStatusWidget"] button svg,
header[data-testid="stHeader"] [data-testid="stAppDeployButton"] svg{
    color:#39ff14!important;
    fill:#39ff14!important;
    stroke:#39ff14!important;
}

/* Stop button */
[data-testid="stStopButton"],
[data-testid="stStopButton"] button {
    background: #07130b !important;
    background-color: #07130b !important;
    border: 1px solid #39ff14 !important;
    color: #fff !important;
}

[data-testid="stStopButton"] svg,
[data-testid="stStopButton"] svg path {
    color: #39ff14 !important;
    fill: #39ff14 !important;
    stroke: #39ff14 !important;
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
        model="openai/gpt-oss-120b"
    )

@st.cache_resource(show_spinner=False)
def load_quiz_model():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        reasoning_effort="low",
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )

embeddings = load_embeddings()
model = load_model()
quiz_model = load_quiz_model()
try:
    cookie_password = st.secrets.get("COOKIE_PASSWORD")
except Exception:
    cookie_password = None

if not cookie_password:
    print("CONFIG ERROR: COOKIE_PASSWORD is missing in .streamlit/secrets.toml")
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

10. Reply in the same language the student used.

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
        print(f"Agent error: {e}")
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
                st.balloons()
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
