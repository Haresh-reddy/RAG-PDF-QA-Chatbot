import time
 
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
 
load_dotenv()  # reads GOOGLE_API_KEY from your .env file
 
# ---------- Settings ----------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64        # chunks embedded per batch (local model, so no API limit)
PAUSE_SECONDS = 0
MAX_CHUNKS = 2000      # safety limit for very large PDFs
TOP_K = 4              # chunks sent to the model
MAX_QUESTION_CHARS = 1000
SHOW_SOURCES = True    # set to False to hide the "Sources" box under answers
 
PROMPT = """Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on this document."
 
Context:
{context}
 
Question:
{question}
 
Answer:"""
 
 
# ---------- Guardrails (replace with the checks from your LLM Guardrails project) ----------
BLOCKED_PHRASES = ["ignore previous instructions", "reveal your system prompt", "jailbreak"]
SENSITIVE_WORDS = ["password", "api key", "api_key"]
 
 
def check_input(text):
    if not text or not text.strip():
        return False, "Please enter a question."
    if len(text) > MAX_QUESTION_CHARS:
        return False, f"Question is too long (max {MAX_QUESTION_CHARS} characters)."
    if any(p in text.lower() for p in BLOCKED_PHRASES):
        return False, "That request looks unsafe, so it was blocked."
    return True, ""
 
 
def check_output(text):
    if not text or not text.strip():
        return False, "The model returned an empty answer."
    if any(w in text.lower() for w in SENSITIVE_WORDS):
        return False, "The answer contained sensitive terms, so it was hidden."
    return True, ""
 
 
# ---------- Helpers ----------
@st.cache_resource
def get_embeddings():
    # Runs on your own computer: free, no rate limit. Downloads ~90 MB the first time.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
 
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
 
 
def build_vector_store(docs, embeddings):
    """Embed ALL chunks in small batches (instead of only the first 20)."""
    store = None
    progress = st.progress(0.0, text="Creating embeddings...")
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        if store is None:
            store = FAISS.from_documents(batch, embeddings)
        else:
            store.add_documents(batch)
        done = min(start + BATCH_SIZE, len(docs))
        progress.progress(done / len(docs), text=f"Embedded {done} of {len(docs)} chunks")
        time.sleep(PAUSE_SECONDS)
    progress.empty()
    return store
 
 
# ---------- Page ----------
st.set_page_config(page_title="PDF Question Answering", page_icon="📄", layout="wide")
st.title("📄 PDF Question Answering System")
st.write("Upload a PDF and ask questions based on its content.")
 
uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])
 
if uploaded_file is None:
    st.info("Upload a PDF to start.")
    st.stop()
 
# Process the PDF only when a NEW file is uploaded (not on every button click)
file_key = (uploaded_file.name, uploaded_file.size)
if st.session_state.get("file_key") != file_key:
    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)
        texts, metadatas = [], []
        for page_no, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
                metadatas.append({"page": page_no})
 
    if not texts:
        st.error("No readable text found. Scanned PDFs need OCR first.")
        st.stop()
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents(texts, metadatas=metadatas)
    truncated = len(docs) > MAX_CHUNKS
    docs = docs[:MAX_CHUNKS]
 
    try:
        store = build_vector_store(docs, get_embeddings())
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        st.stop()
 
    st.session_state.update(file_key=file_key, store=store, n_chunks=len(docs), truncated=truncated, history=[], last=None)
 
st.sidebar.success(f"PDF processed: {st.session_state['n_chunks']} chunks stored in FAISS.")
if st.session_state["truncated"]:
    st.sidebar.warning(f"This PDF is large, so only the first {MAX_CHUNKS} chunks were used.")
 
# Chat box stays at the bottom of the page
question = st.chat_input("Ask a question about your PDF")
 
if question:
    sources = []
    ok, reason = check_input(question)
    if not ok:
        answer = reason
    else:
        results = st.session_state["store"].similarity_search(question, k=TOP_K)
        context = "\n\n".join(r.page_content for r in results)
        try:
            answer = get_llm().invoke(PROMPT.format(context=context, question=question)).content
            ok_out, reason_out = check_output(answer)
            if ok_out:
                # Don't show unrelated chunks as "sources" when the model says it doesn't know
                if not answer.strip().lower().startswith("i don't know"):
                    sources = [(r.metadata.get("page", "?"), r.page_content) for r in results]
            else:
                answer = reason_out
        except Exception as e:
            answer = f"Model error: {e}"
    st.session_state["history"].append((question, answer))
    st.session_state["last"] = {"question": question, "answer": answer, "sources": sources}
 
# Main area: only the latest question and answer. Older ones are in the sidebar.
last = st.session_state.get("last")
if last:
    with st.chat_message("user"):
        st.markdown(last["question"])
    with st.chat_message("assistant"):
        st.markdown(last["answer"])
        if SHOW_SOURCES and last["sources"]:
            with st.expander("Sources"):
                for page, text in last["sources"]:
                    st.markdown(f"**Page {page}**")
                    st.caption(text)
 
# Sidebar: chat history (drawn last so it includes the newest question)
with st.sidebar:
    st.divider()
    st.subheader("Questions asked")
    if st.session_state["history"]:
        for i, (q, a) in enumerate(st.session_state["history"], start=1):
            with st.expander(f"{i}. {q}"):
                st.write(a)
    else:
        st.caption("No questions yet.")
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.session_state["last"] = None
        st.rerun()
 