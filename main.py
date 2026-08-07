from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException
from Youtube_Chat_Bot.config import rag_config_file
from Youtube_Chat_Bot.logging.logger import logging 
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation

if __name__ == "__main__":
    logging.info("RAG Indexing started")

    indexing_obj = indexing(rag_config_file.video_id,rag_config_file.chunk_size,rag_config_file.chunk_overlap,rag_config_file.embedding_model_name)
    vector_store = indexing_obj.vector_store()
    
    logging.info("RAG Generation started")

    chatbot = generation(vector_store,rag_config_file.search_type,rag_config_file.search_kwargs,rag_config_file.repo_id,rag_config_file.task)

    print("Ask question based on video")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Bot: Bye ")
            break

        response = chatbot.chat(query)
        print("Bot:", response)
