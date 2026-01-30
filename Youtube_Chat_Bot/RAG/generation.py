import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()


class generation:
    def __init__(self, vector_store, search_type, search_kwargs, repo_id, task):
        self.vector_store = vector_store
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.repo_id = repo_id
        self.task = task

        self.memory = [
            SystemMessage(content="You are a helpful assistant. Answer only from the transcript.")
        ]

    def retriever(self):
        return self.vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(self):
        history = ""
        for msg in self.memory:
            if isinstance(msg, HumanMessage):
                history += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history += f"Assistant: {msg.content}\n"
        return history

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
                    If question ask in english so give the answer in english or
                    If question ask in hindi so give the answer in hindi or
                    If question ask in hinglish so give the answer in hinglish

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

    def chat(self, question):
        self.memory.append(HumanMessage(content=question))
        answer = self.chain().invoke(question)
        self.memory.append(AIMessage(content=answer))
        return answer
