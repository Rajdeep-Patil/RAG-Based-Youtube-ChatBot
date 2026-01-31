import sys
import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException
from Youtube_Chat_Bot.logging.logger import logging

load_dotenv()


class indexing:
    def __init__(self, video_id, chunk_size, chunk_overlap, embedding_model_name):
        self.video_id = video_id
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

        logging.info(f"Indexing initialized for video_id={video_id}")

    def youtube_transcript(self):
        logging.info("Fetching YouTube transcript...")
        yt_api = YouTubeTranscriptApi()

        try:
            video_transcript = yt_api.fetch(
                video_id=self.video_id,
                languages=['en', 'hi']
            )

            transcript = " ".join(text.text for text in video_transcript)
            logging.info(f"Transcript fetched successfully | Length: {len(transcript)} chars")

            return transcript

        except Exception as e:
            logging.error("Failed to fetch YouTube transcript", exc_info=True)
            raise YoutubeChatBotException(e, sys)

    def text_splitter(self):
        logging.info("Splitting transcript into chunks...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        transcript = self.youtube_transcript()
        docs = splitter.create_documents(
            [transcript],
            metadatas=[{"video_id": self.video_id}]
        )

        logging.info(f"Text split completed | Total chunks: {len(docs)}")
        return docs

    def get_vectorstore(self):
        logging.info("Connecting to Pinecone...")

        try:
            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

            logging.info("Pinecone connection successful")

            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )

            logging.info(f"Embedding model loaded: {self.embedding_model_name}")

            vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="page_content"
            )

            return vectorstore

        except Exception as e:
            logging.error("Failed to initialize Pinecone VectorStore", exc_info=True)
            raise YoutubeChatBotException(e, sys)

    def vector_store(self):
        logging.info("Starting vector store indexing pipeline...")

        try:
            docs = self.text_splitter()
            vectorstore = self.get_vectorstore()

            logging.info("Adding documents to Pinecone...")
            vectorstore.add_documents(docs)

            logging.info("Documents successfully indexed in Pinecone")
            return vectorstore

        except Exception as e:
            logging.error("Vector store indexing failed", exc_info=True)
            raise YoutubeChatBotException(e, sys)