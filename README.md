PDF Question Answering Chatbot (RAG)

Upload a PDF and ask questions about it. The app finds the most relevant passages in the document, sends only those to Google Gemini, and shows the answer with the pages it came from. If the answer is not in the document, it says so instead of guessing.

Live demo: add your Streamlit link here

Screenshots

Answer with sources I don't know reply Guardrail blocking an unsafe prompt Sidebar with question history

Features
Upload any text-based PDF
Answers only from the document, with page-number sources
Replies "I don't know based on this document." when the answer is not in the PDF
Chat history that stays on screen
Embeddings and the vector index are built once per upload, not on every question
Input and output guardrails: blocks prompt-injection phrases and over-long or empty questions, and hides empty or sensitive answers
How it works
Read: pypdf extracts the text of each page.
Split: LangChain's RecursiveCharacterTextSplitter cuts the text into chunks (1000 characters, 200 overlap) and keeps the page number.
Embed: Hugging Face sentence-transformers/all-MiniLM-L6-v2 turns each chunk into a vector. It runs locally, so there is no API rate limit.
Store and search: vectors are stored in a FAISS index; the question is embedded and the 4 most similar chunks are retrieved.
Answer: Gemini (gemini-2.5-flash) answers using only those chunks.
Guardrails: simple checks run before the question reaches the model and after the answer comes back.
Tech stack

Python, Streamlit, LangChain, FAISS, Hugging Face sentence-transformers, Google Gemini API, pypdf

Run locally
bash
git clone https://github.com/Haresh-reddy/RAG-PDF-QA-Chatbot.git
cd RAG-PDF-QA-Chatbot
pip install -r requirements.txt

Create a .env file in the project folder:

GOOGLE_API_KEY=your-key-here

Get a free key from Google AI Studio. Never commit your key to GitHub.

bash
streamlit run streamlit_app.py

The first run downloads the embedding model (about 90 MB).

Deploy on Streamlit Community Cloud
Push this repo to GitHub (without .env).
On share.streamlit.io, create a new app from the repo and choose streamlit_app.py.
In the app's Settings, then Secrets, add: GOOGLE_API_KEY = "your-key-here"
Tested with

The official Python tutorial PDF (433 chunks). Example checks:

A question from early in the PDF (lists) and from near the end (virtual environments) were both answered correctly.
A question outside the document returned "I don't know based on this document."
A prompt-injection attempt was blocked.
Limitations
Scanned (image-only) PDFs need OCR first.
The guardrails are simple keyword and length checks, not a complete safety system.
Answers can only be as good as the retrieved chunks.
The free Gemini tier limits how many questions you can ask per minute.
Possible improvements
Support several PDFs at once
Add a similarity cut-off so weak matches are not sent to the model
Compare different chunk sizes and top-k values
