import re
import streamlit as st
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation
from Youtube_Chat_Bot.config import rag_config_file


st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube Video Chatbot")
st.caption("Ask questions from a YouTube video transcript")


# ---- Session state ----
st.session_state.setdefault("VECTOR_CACHE", {})
st.session_state.setdefault("CHAT_CACHE", {})
st.session_state.setdefault("MESSAGES", {})


def extract_video_id(url: str):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def ensure_pipeline_ready(video_id: str):
    """Build the vector store + chat pipeline for a video_id if not cached.
    Cleans up partial state on failure so a retry doesn't crash later."""
    if video_id in st.session_state.CHAT_CACHE:
        return st.session_state.CHAT_CACHE[video_id]

    try:
        index = indexing(
            video_id=video_id,
            chunk_size=rag_config_file.chunk_size,
            chunk_overlap=rag_config_file.chunk_overlap,
            embedding_model_name=rag_config_file.embedding_model_name
        )

        vector_store = index.vector_store()
        st.session_state.VECTOR_CACHE[video_id] = vector_store

        bot = generation(
            vector_store=vector_store,
            search_type=rag_config_file.search_type,
            search_kwargs=rag_config_file.search_kwargs,
            repo_id=rag_config_file.repo_id,
            task=rag_config_file.task
        )
        st.session_state.CHAT_CACHE[video_id] = bot

        return bot

    except Exception:
        # Roll back any partial state so a retry starts clean
        st.session_state.VECTOR_CACHE.pop(video_id, None)
        st.session_state.CHAT_CACHE.pop(video_id, None)
        raise


# ---- Video input ----
video_url = st.text_input("Paste YouTube Video URL")

video_id = None
if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.video(video_url)
        st.session_state.MESSAGES.setdefault(video_id, [])
    else:
        st.error("Invalid YouTube URL")
else:
    st.info("Paste a YouTube URL above to get started.")


# ---- Chat history ----
if video_id and video_id in st.session_state.MESSAGES:
    for msg in st.session_state.MESSAGES[video_id]:
        st.chat_message(msg["role"]).write(msg["content"])


# ---- Chat input ----
question = st.chat_input(
    "Ask a question about the video",
    disabled=video_id is None
)

if question and video_id:
    st.session_state.MESSAGES[video_id].append(
        {"role": "user", "content": question}
    )
    st.chat_message("user").write(question)

    try:
        with st.spinner("Fetching transcript & indexing..."):
            bot = ensure_pipeline_ready(video_id)
            answer = bot.chat(question)

        st.session_state.MESSAGES[video_id].append(
            {"role": "assistant", "content": answer}
        )
        st.chat_message("assistant").write(answer)

    except Exception:
        st.error(
            "⚠️ Transcript not available for this video.\n\n"
            "Possible reasons:\n"
            "- Captions disabled\n"
            "- YouTube blocked cloud IP\n"
            "- Private / restricted video"
        )


# ---- Clear chat ----
if video_id:
    if st.button("🗑️ Clear Chat"):
        st.session_state.MESSAGES[video_id] = []
        st.rerun()