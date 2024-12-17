from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import os
import logging
import pandas as pd
import numpy as np
from openai import AzureOpenAI

# Initialize Flask application and specify the folder for HTML templates
app = Flask(__name__, template_folder='templates')

# Initialize Qdrant client for storing and searching embeddings
qdrant_client = QdrantClient(host="localhost", port=6333)  # Qdrant server connection details
COLLECTION_NAME = "knowledge_base"  # Name of the Qdrant collection

# Configure Rate Limiting to prevent abuse
limiter = Limiter(
    get_remote_address,  # Use client IP address for rate limiting
    app=app,  # Bind to the Flask app
    default_limits=["10 per minute"]  # Allow max 10 requests per minute per IP
)

# Configure Logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)

# Load environment variables from the .env file
load_dotenv()

# Azure OpenAI Configuration: Load API keys and endpoints from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://default-endpoint.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
COMPLETION_DEPLOYMENT_ID = os.getenv("COMPLETION_DEPLOYMENT_ID", "default-completion-deployment")
EMBEDDING_DEPLOYMENT_ID = os.getenv("EMBEDDING_DEPLOYMENT_ID", "default-embedding-deployment")

# Validate critical configurations
if not AZURE_OPENAI_API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY is not set in the environment.")

# Initialize Azure OpenAI client for generating embeddings and completions
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01",  # Specific version of Azure OpenAI API
    azure_endpoint=AZURE_OPENAI_ENDPOINT  # Azure OpenAI endpoint
)

# Function to load knowledge base from CSV and precompute embeddings
def load_knowledge_base(csv_file):
    """
    Loads the knowledge base from a CSV file, generates embeddings for each question,
    and stores the embeddings in Qdrant.
    
    Args:
        csv_file (str): Path to the CSV file containing 'question' and 'answer' columns.
    """
    try:
        # Load CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns.")

        # Convert DataFrame rows into a list of dictionaries
        knowledge_base = df.to_dict(orient='records')

        # Check if Qdrant collection exists, create it if it doesn't
        if COLLECTION_NAME not in [collection.name for collection in qdrant_client.get_collections().collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(size=1536, distance="Cosine"),  # Define vector size and similarity metric
            )

        # Generate embeddings for each question and upsert into Qdrant
        for idx, entry in enumerate(knowledge_base):
            question = entry['question']

            # Generate embedding for the question using Azure OpenAI
            response = client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT_ID,
                input=question
            )
            embedding = response.data[0].embedding  # Extract the embedding vector

            # Store the embedding and payload in Qdrant
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=idx,  # Unique ID for the point
                        vector=embedding,  # Embedding vector
                        payload={"question": question, "answer": entry['answer']}  # Store question and answer
                    )
                ]
            )
        print("Knowledge base successfully loaded into Qdrant.")
    except Exception as e:
        raise RuntimeError(f"Error loading knowledge base: {e}")

# Function to retrieve relevant passages from Qdrant
def retrieve_relevant_passages(query, top_n=1, min_score=0.95):
    """
    Retrieves the most relevant passages from Qdrant using cosine similarity search.

    Args:
        query (str): The user's query.
        top_n (int): Number of top relevant passages to retrieve.
        min_score (float): Minimum similarity score threshold for relevance.

    Returns:
        list: List of relevant questions and answers.
    """
    try:
        # Generate embedding for the query
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_ID,
            input=query
        )
        query_embedding = response.data[0].embedding  # Extract query embedding

        # Perform similarity search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,  # Use the query embedding for search
            limit=top_n  # Number of results to retrieve
        )

        # Extract questions, answers, and scores from search results
        relevant_passages = [
            {
                "question": result.payload["question"],
                "answer": result.payload["answer"],
                "score": result.score
            }
            for result in search_results if result.score >= min_score
        ]

        return relevant_passages
    except Exception as e:
        raise RuntimeError(f"Error retrieving relevant passages: {e}")

# Function to format the final response using GPT-4o
def format_response_with_gpt4o(query, relevant_passages):
    """
    Formats the retrieved relevant passages into a human-readable response using GPT-4o.

    Args:
        query (str): User's query.
        relevant_passages (list): Relevant passages retrieved from the knowledge base.

    Returns:
        str: Formatted, human-friendly response.
    """
    try:
        context = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in relevant_passages])
        prompt = (
            f"The user asked: {query}\n\n"
            f"Here are the relevant FAQs:\n{context}\n\n"
            f"Please summarize and format this information into a concise, human-friendly response. Act like you respond directly , beginning with Dear user, here is your answer. If passage context and your answer are very different, answer sorry i cant't help you. Do not answer!!!"
        )
        
        completion = client.chat.completions.create(
            model=COMPLETION_DEPLOYMENT_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear and concise responses."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error formatting response with GPT-4o: {e}")

# Flask endpoint: '/ask' to handle user queries
@app.route('/ask', methods=['POST'])
@limiter.limit("10/minute")  # Apply rate limiting to this endpoint
def ask():
    """
    Handles user queries: retrieves relevant passages, formats them using GPT-4o, and returns the response.
    """
    try:
        # Parse JSON input from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query not provided"}), 400  # Return error if query is missing

        query = data['query']

        # Step 1: Retrieve relevant passages from Qdrant
        relevant_passages = retrieve_relevant_passages(query, top_n=3, min_score=0.8)

        # If no relevant passages are found, return a strict response
        if not relevant_passages:
            return jsonify({
                "query": query,
                "relevant_passages": [],
                "formatted_response": "Sorry, I could not find any relevant information."
            })

        # Step 2: Format the response using GPT-4o
        formatted_response = format_response_with_gpt4o(query, relevant_passages)

        # Return the formatted response
        return jsonify({
            "query": query,
            "relevant_passages": relevant_passages,
            "formatted_response": formatted_response
        })
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Flask endpoint: '/health' to check the application health
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Flask endpoint: Home route to render the main page
@app.route('/')
def home():
    """
    Renders the HTML home page for the application.
    """
    return render_template('index.html')

if __name__ == "__main__":
    try:
        # Pre-load the knowledge base and precompute embeddings
        load_knowledge_base("bank_faq_questions_answers.csv")
        # Start the Flask server
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start the application: {e}")
