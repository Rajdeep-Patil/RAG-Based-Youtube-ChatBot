import re
import streamlit as st
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube RAG Chatbot")
st.caption("Ask questions directly from any YouTube video")

# ---------------- CACHE ----------------
if "VECTOR_CACHE" not in st.session_state:
    st.session_state.VECTOR_CACHE = {}

if "CHAT_CACHE" not in st.session_state:
    st.session_state.CHAT_CACHE = {}

# ---------------- FUNCTIONS ----------------
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


# ---------------- UI ----------------
video_url = st.text_input("📌 Paste YouTube URL")

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.video(video_url)
    else:
        st.error("❌ Invalid YouTube URL")

question = st.chat_input("Ask a question from the video")

# ---------------- CHAT LOGIC ----------------
if question and video_url:
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        with st.spinner("Indexing video & thinking..."):
            # Create vector store once per video
            if video_id not in st.session_state.VECTOR_CACHE:
                index = indexing(
                    video_id=video_id,
                    chunk_size=1000,
                    chunk_overlap=200,
                    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.VECTOR_CACHE[video_id] = index.vector_store()

                st.session_state.CHAT_CACHE[video_id] = generation(
                    vector_store=st.session_state.VECTOR_CACHE[video_id],
                    search_type="similarity",
                    search_kwargs={"k": 4},
                    repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
                    task="text-generation"
                )

            bot = st.session_state.CHAT_CACHE[video_id]
            answer = bot.chat(question)

        # Chat UI
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)
