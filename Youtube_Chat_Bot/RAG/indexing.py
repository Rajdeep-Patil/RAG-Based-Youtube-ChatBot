import sys
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException


class indexing:
    def __init__(self, video_id, chunk_size, chunk_overlap, embedding_model_name):
        self.video_id = video_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

    def youtube_transcript(self):
        yt_api = YouTubeTranscriptApi()
        try:
            video_transcript = yt_api.fetch(video_id=self.video_id,languages=['en','hi'])
            transcript = " ".join(text.text for text in video_transcript)
            return transcript
        except Exception as e:
            raise YoutubeChatBotException(e, sys)

    def text_splitter(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        transcript = self.youtube_transcript()
        return splitter.create_documents([transcript])

    def embedding_model(self):
        try:
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        except Exception as e:
            raise YoutubeChatBotException(e, sys)

    def vector_store(self):
        embeddings = self.embedding_model()
        docs = self.text_splitter()
        return FAISS.from_documents(docs, embeddings)
