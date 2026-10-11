# Transcript
video_id = 'd2kxUVwWWwU'

# Splitter 
chunk_size = 1000
chunk_overlap = 200

# Embedding Model
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
text_key = 'text'

# Retriever
search_type='similarity'
search_kwargs={'k':4}

# Chatmodel
repo_id='meta-llama/Llama-3.2-3B-Instruct'
task='text-generation'