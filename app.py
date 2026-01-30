import re
from flask import Flask, render_template, request, jsonify
from Youtube_Chat_Bot.RAG.indexing import indexing
from Youtube_Chat_Bot.RAG.generation import generation

app = Flask(__name__)

VECTOR_CACHE = {}
CHAT_CACHE = {}


def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    video_url = data.get("video_url")
    question = data.get("question")

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"answer": "❌ Invalid YouTube URL"})

    if video_id not in VECTOR_CACHE:
        index = indexing(
            video_id=video_id,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        VECTOR_CACHE[video_id] = index.vector_store()

        CHAT_CACHE[video_id] = generation(
            vector_store=VECTOR_CACHE[video_id],
            search_type="similarity",
            search_kwargs={"k": 4},
            repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
            task="text-generation"
        )

    bot = CHAT_CACHE[video_id]
    answer = bot.chat(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=True)
