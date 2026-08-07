import os
import sys
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException
from Youtube_Chat_Bot.logging.logger import logging
from dotenv import load_dotenv
load_dotenv()
class generation:
    def __init__(self, vector_store, search_type, search_kwargs, repo_id, task):
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.repo_id = repo_id
        self.task = task

        logging.info(f"Generation initialized with repo_id={repo_id}, task={task}")

    def retriever(self):
        logging.info(f"Creating retriever with search_type={self.search_type} and search_kwargs={self.search_kwargs}")
        try:
            retriever = self.vector_store.as_retriever(
                search_type=self.search_type,
                search_kwargs=self.search_kwargs
            )
            logging.info("Retriever created successfully")
            return retriever
        except Exception as e:
            logging.error("Failed to create retriever", exc_info=True)
            raise YoutubeChatBotException(e, sys)

    def format_docs(self, docs):
        logging.info(f"Formatting {len(docs)} documents into context")
        return "\n\n".join(doc.page_content for doc in docs)

    def chain(self):
        logging.info("Building HuggingFace chain...")
        try:
            llm = HuggingFaceEndpoint(
                repo_id=self.repo_id,
                task=self.task,
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
            )

            model = ChatHuggingFace(llm=llm)
            logging.info("HuggingFace model loaded successfully")

            prompt = PromptTemplate(
            template="""
            You are a helpful assistant answering questions about a YouTube video transcript.
            - If the user greets you or makes small talk, respond naturally and briefly.
            - If the user asks a question about the video and the answer isn't in the provided context, say you don't know.
            Context: {context}
            Question: {question}""",
            input_variables=["context", "question"]
            )

            parallel = RunnableParallel({
                "context": self.retriever() | RunnableLambda(self.format_docs),
                "question": RunnablePassthrough()
            })

            logging.info("Chain built successfully")
            return parallel | prompt | model | StrOutputParser()

        except Exception as e:
            logging.error("Failed to build chain", exc_info=True)
            raise YoutubeChatBotException(e, sys)

    def chat(self, question):
        logging.info(f"Received user question: {question}")

        try:
            logging.info("Invoking chain to generate answer...")
            answer = self.chain().invoke(question)
            logging.info(f"Answer generated: {answer}")

            return answer

        except Exception as e:
            logging.error("Failed during chat invocation", exc_info=True)
            raise YoutubeChatBotException(e, sys)