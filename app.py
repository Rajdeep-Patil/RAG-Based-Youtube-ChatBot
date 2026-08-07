import re
import streamlit as st
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation
from Youtube_Chat_Bot.config import rag_config_file

st.title("🎥 YouTube Video Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    st.session_state.bot = None

def extract_video_id(url):
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

video_url = st.text_input("Paste YouTube Video URL")

if video_url:
    video_id = extract_video_id(video_url)

    if video_id and st.session_state.bot is None:
        with st.spinner("Indexing video..."):
            idx = indexing(
                video_id=video_id,
                chunk_size=rag_config_file.chunk_size,
                chunk_overlap=rag_config_file.chunk_overlap,
                embedding_model_name=rag_config_file.embedding_model_name
            )
            vector_store = idx.vector_store()

            st.session_state.bot = generation(
                vector_store=vector_store,
                search_type=rag_config_file.search_type,
                search_kwargs=rag_config_file.search_kwargs,
                repo_id=rag_config_file.repo_id,
                task=rag_config_file.task
            )

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("Ask a question about the video")

if question and st.session_state.bot:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    answer = st.session_state.bot.chat(question)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)