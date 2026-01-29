from Youtube_Chat_Bot.exception.exception import YoutubeChatBotException
from Youtube_Chat_Bot.constant import training_pipeline
from Youtube_Chat_Bot.logging.logger import logging 
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation

if __name__ == "__main__":
    logging.info("RAG Indexing started")

    indexing_obj = indexing(
        training_pipeline.video_id,
        training_pipeline.chunk_size,
        training_pipeline.chunk_overlap,
        training_pipeline.embedding_model_name
    )

    vector_store = indexing_obj.vector_store()
    
    logging.info("RAG Generation started")

    chatbot = generation(
        vector_store,
        training_pipeline.search_type,
        training_pipeline.search_kwargs,
        training_pipeline.repo_id,
        training_pipeline.task
    )

    print("Ask question based on video")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Bot: Bye ")
            break

        response = chatbot.chat(query)
        print("Bot:", response)
