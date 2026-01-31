import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException
from Youtube_Chat_Bot.logging.logger import logging

load_dotenv()
class generation:
    def __init__(self, vector_store, search_type, search_kwargs, repo_id, task):
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.repo_id = repo_id
        self.task = task

        self.memory = [
            SystemMessage(
                content="You are a helpful assistant. Answer only from the transcript."
            )
        ]

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

    def format_history(self):
        history = ""
        for msg in self.memory:
            if isinstance(msg, HumanMessage):
                history += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"Assistant: {msg.content}\n"
        logging.debug(f"Formatted chat history:\n{history}")
        return history

    def trim_memory(self, max_turns=10):
        logging.info(f"Trimming memory to last {max_turns} turns")
        system_msg = self.memory[0]
        convo = self.memory[1:]  
        max_messages = max_turns * 2  

        if len(convo) > max_messages:
            convo = convo[-max_messages:]

        self.memory = [system_msg] + convo
        logging.debug(f"Memory length after trimming: {len(self.memory)} messages")

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
                    You are a helpful assistant that answers questions strictly using the provided transcript.

                        Rules:
                        1. Use ONLY the information present in the Transcript.
                        2. Do NOT use outside knowledge or make assumptions.
                        3. If the answer is NOT present in the transcript, respond with:
                        "I don't know." 
                        Then briefly explain why the transcript does not contain this information.
                        4. Match the language of the user's question:
                        - English → English
                        - Hindi → Hindi
                        - Hinglish → Hinglish
                        5. Keep answers clear, concise, and factual.

                        Chat History:
                        {history}

                        Transcript Context:
                        {context}

                        User Question:
                        {question}""",
                input_variables=["history", "context", "question"]
            )

            parallel = RunnableParallel({
                "context": self.retriever() | RunnableLambda(self.format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: self.format_history())
            })

            logging.info("Chain built successfully")
            return parallel | prompt | model | StrOutputParser()

        except Exception as e:
            logging.error("Failed to build chain", exc_info=True)
            raise YoutubeChatBotException(e, sys)

    def chat(self, question):
        logging.info(f"Received user question: {question}")

        if question.lower().strip() in ["exit", "quit", "clear"]:
            self.memory = [self.memory[0]]
            logging.info("Memory cleared")
            return "Memory cleared."

        self.memory.append(HumanMessage(content=question))
        self.trim_memory(max_turns=10)

        try:
            logging.info("Invoking chain to generate answer...")
            answer = self.chain().invoke(question)
            logging.info(f"Answer generated: {answer}")

            self.memory.append(AIMessage(content=answer))
            self.trim_memory(max_turns=10)

            return answer

        except Exception as e:
            logging.error("Failed during chat invocation", exc_info=True)
            raise YoutubeChatBotException(e, sys)
