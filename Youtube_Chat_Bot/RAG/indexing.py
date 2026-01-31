import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException

load_dotenv()


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
        return splitter.create_documents(
            [transcript],
            metadatas=[{"video_id": self.video_id}]
        )

    def get_vectorstore(self):
        # ✅ Pinecone SDK client
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        # ✅ Connect to existing index
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        # ✅ Correct LangChain wrapper
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="page_content"
        )

        return vectorstore

    def vector_store(self):
        docs = self.text_splitter()
        vectorstore = self.get_vectorstore()
        vectorstore.add_documents(docs)
        return vectorstore
