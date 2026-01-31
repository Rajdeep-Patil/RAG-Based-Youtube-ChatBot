import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


class generation:
    def __init__(self, vector_store, search_type, search_kwargs, repo_id, task):
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.repo_id = repo_id
        self.task = task

        # System message (always preserved)
        self.memory = [
            SystemMessage(
                content="You are a helpful assistant. Answer only from the transcript."
            )
        ]

    # ---------------- RETRIEVER ----------------
    def retriever(self):
        return self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs
        )

    # ---------------- FORMAT DOCS ----------------
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ---------------- FORMAT HISTORY ----------------
    def format_history(self):
        history = ""
        for msg in self.memory:
            if isinstance(msg, HumanMessage):
                history += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"Assistant: {msg.content}\n"
        return history

    # ---------------- MEMORY LIMIT ----------------
    def trim_memory(self, max_turns=10):
        """
        Keeps only last `max_turns` human-ai conversations.
        SystemMessage is always preserved.
        """
        system_msg = self.memory[0]
        convo = self.memory[1:]  # remove system message

        max_messages = max_turns * 2  # Human + AI

        if len(convo) > max_messages:
            convo = convo[-max_messages:]

        self.memory = [system_msg] + convo

    # ---------------- CHAIN ----------------
    def chain(self):
        llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            task=self.task,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        model = ChatHuggingFace(llm=llm)

        prompt = PromptTemplate(
            template="""
                Answer ONLY using the transcript context.
                If the answer is not in the transcript, say "I don't know, and also explain the reason".
                If the question is in English, answer in English.
                If the question is in Hindi, answer in Hindi.
                If the question is in Hinglish, answer in Hinglish.

                Chat History:
                {history}

                Transcript:
                {context}

                Question:
                {question}
                """,
            input_variables=["history", "context", "question"]
        )

        parallel = RunnableParallel({
            "context": self.retriever() | RunnableLambda(self.format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: self.format_history())
        })

        return parallel | prompt | model | StrOutputParser()

    # ---------------- CHAT ----------------
    def chat(self, question):
        # Handle commands (optional but safe)
        if question.lower().strip() in ["exit", "quit", "clear"]:
            self.memory = [self.memory[0]]
            return "🔄 Memory cleared."

        # Add user message
        self.memory.append(HumanMessage(content=question))

        # Trim before LLM call
        self.trim_memory(max_turns=10)

        # Get answer
        answer = self.chain().invoke(question)

        # Add assistant message
        self.memory.append(AIMessage(content=answer))

        # Trim after LLM call
        self.trim_memory(max_turns=10)

        return answer
