from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import os
import glob
import logging
import uuid
from openai import AzureOpenAI

# Initialize Flask application and specify the folder for HTML templates
app = Flask(__name__, template_folder='templates')

# Initialize Qdrant client for storing and searching embeddings
qdrant_client = QdrantClient(host="host.docker.internal", port=6333)
COLLECTION_NAME = "washing_machine_knowledge"  # Name of Qdrant collection

# Configure Rate Limiting
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://default-endpoint.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
COMPLETION_DEPLOYMENT_ID = os.getenv("COMPLETION_DEPLOYMENT_ID", "default-completion-deployment")
EMBEDDING_DEPLOYMENT_ID = os.getenv("EMBEDDING_DEPLOYMENT_ID", "default-embedding-deployment")

if not AZURE_OPENAI_API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY is not set.")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Function to load knowledge base from multiple text files in a folder
def load_knowledge_base_from_folder(folder_path):
    try:
        # Get a list of all .txt files in the folder
        txt_files = glob.glob(f"{folder_path}/*.txt")
        if not txt_files:
            raise FileNotFoundError("No .txt files found in the specified folder.")

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Split raw text into passages (split on double newlines for paragraphs)
            passages = [p.strip() for p in raw_content.split("\n\n") if p.strip()]

            # Create Qdrant collection if it doesn't exist
            if COLLECTION_NAME not in [collection.name for collection in qdrant_client.get_collections().collections]:
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(size=1536, distance="Cosine")
                )

            # Generate embeddings for each passage
            for passage in passages:
                response = client.embeddings.create(
                    model=EMBEDDING_DEPLOYMENT_ID,
                    input=passage
                )
                embedding = response.data[0].embedding

                # Generate a unique UUID for the passage ID
                point_id = str(uuid.uuid4())

                # Upsert into Qdrant
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        qdrant_models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={"text": passage, "source": txt_file}
                        )
                    ]
                )
        print("Knowledge base successfully loaded from text files.")
    except Exception as e:
        raise RuntimeError(f"Error loading knowledge base: {e}")

# Function to retrieve relevant passages
def retrieve_relevant_passages(query, top_n=3, min_score=0.8):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_ID,
            input=query
        )
        query_embedding = response.data[0].embedding

        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_n
        )

        return [
            {
                "text": result.payload["text"],
                "score": result.score
            }
            for result in search_results if result.score >= min_score
        ]
    except Exception as e:
        raise RuntimeError(f"Error retrieving relevant passages: {e}")

# Function to format response with GPT-4o
def format_response_with_gpt4o(query, relevant_passages):
    try:
        context = "\n\n".join([entry['text'] for entry in relevant_passages])
        prompt = (
            f"The user asked: {query}\n\n"
            f"Here is some relevant information:\n\n{context}\n\n"
            f"Please summarize and format this into a helpful response. "
            f"Start with 'Dear user, here is your answer'. "
            f"If no relevant information is found, say 'Sorry, I can't help with that.'"
        )

        completion = client.chat.completions.create(
            model=COMPLETION_DEPLOYMENT_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in product details."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error formatting response: {e}")

# Flask endpoint: '/ask'
@app.route('/ask', methods=['POST'])
@limiter.limit("10/minute")
def ask():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query not provided"}), 400

        query = data['query']

        # Retrieve relevant passages
        relevant_passages = retrieve_relevant_passages(query, top_n=3, min_score=0.8)

        if not relevant_passages:
            return jsonify({
                #"query": query,
                #"relevant_passages": [],
                "formatted_response": "Sorry, I could not find any relevant information."
            })

        # Format response with GPT-4o
        formatted_response = format_response_with_gpt4o(query, relevant_passages)

        return jsonify({
            #"query": query,
            #"relevant_passages": relevant_passages,
            "formatted_response": formatted_response
        })
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Flask endpoint: '/health'
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Home route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    try:
        # Load the knowledge base from raw text
        load_knowledge_base_from_folder("documents")
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start the application: {e}")
