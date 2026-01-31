import re
import streamlit as st
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation
from Youtube_Chat_Bot.constant import training_pipeline


st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube Video Chatbot")
st.caption("Ask questions from a YouTube video transcript")


if "VECTOR_CACHE" not in st.session_state:
    st.session_state.VECTOR_CACHE = {}

if "CHAT_CACHE" not in st.session_state:
    st.session_state.CHAT_CACHE = {}

if "MESSAGES" not in st.session_state:
    st.session_state.MESSAGES = {}


def extract_video_id(url: str):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


video_url = st.text_input("Paste YouTube Video URL")

video_id = None
if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.video(video_url)

        if video_id not in st.session_state.MESSAGES:
            st.session_state.MESSAGES[video_id] = []
    else:
        st.error("Invalid YouTube URL")


if video_id and video_id in st.session_state.MESSAGES:
    for msg in st.session_state.MESSAGES[video_id]:
        st.chat_message(msg["role"]).write(msg["content"])


question = st.chat_input("Ask a question about the video")


if question and video_id:
    try:
        st.session_state.MESSAGES[video_id].append(
            {"role": "user", "content": question}
        )

        st.chat_message("user").write(question)

        with st.spinner("Fetching transcript & indexing..."):
            if video_id not in st.session_state.VECTOR_CACHE:
                index = indexing(
                    video_id=video_id,
                    chunk_size=training_pipeline.chunk_size,
                    chunk_overlap=training_pipeline.chunk_overlap,
                    embedding_model_name=training_pipeline.embedding_model_name
                )

                st.session_state.VECTOR_CACHE[video_id] = index.vector_store()

                st.session_state.CHAT_CACHE[video_id] = generation(
                    vector_store=st.session_state.VECTOR_CACHE[video_id],
                    search_type=training_pipeline.search_type,
                    search_kwargs=training_pipeline.search_kwargs,
                    repo_id=training_pipeline.repo_id,
                    task=training_pipeline.task
                )

            bot = st.session_state.CHAT_CACHE[video_id]
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


if video_id:
    if st.button("🗑️ Clear Chat"):
        st.session_state.MESSAGES[video_id] = []
        st.rerun()
