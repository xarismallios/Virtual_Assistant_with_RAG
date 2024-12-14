from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import os
import logging
from openai import AzureOpenAI
import pandas as pd
import numpy as np

# Initialize Flask application and specify the folder for templates
app = Flask(__name__, template_folder='templates')

# Configure Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"]
)

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file in local development
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://default-endpoint.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
COMPLETION_DEPLOYMENT_ID = os.getenv("COMPLETION_DEPLOYMENT_ID", "default-completion-deployment")
EMBEDDING_DEPLOYMENT_ID = os.getenv("EMBEDDING_DEPLOYMENT_ID", "default-embedding-deployment")

# Validate critical configurations
if not AZURE_OPENAI_API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY is not set in the environment.")

# Set environment variables for deployment details dynamically
os.environ["MODEL_DEPLOYMENT_NAME"] = COMPLETION_DEPLOYMENT_ID
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
# Initialize the Azure OpenAI client for API interactions
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Load the knowledge base into memory and generate embeddings
knowledge_base = []
knowledge_base_embeddings = None

def load_knowledge_base(csv_file):
    """
    Loads the knowledge base from a CSV file and computes embeddings for each question.

    Args:
        csv_file (str): Path to the CSV file containing 'question' and 'answer' columns.

    Raises:
        ValueError: If required columns are missing in the CSV.
    """
    global knowledge_base, knowledge_base_embeddings
    try:
        df = pd.read_csv(csv_file)
        if 'question' not in df.columns or 'answer' not in df.columns:
            raise ValueError("CSV must have 'question' and 'answer' columns.")

        # Convert the knowledge base into a list of dictionaries for easier access
        knowledge_base = df.to_dict(orient='records')

        # Generate embeddings for all questions in the knowledge base
        questions = [entry['question'] for entry in knowledge_base]
        knowledge_base_embeddings = []
        for question in questions:
            response = client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT_ID,
                input=question
            )
            # Store embeddings for each question
            knowledge_base_embeddings.append(response.data[0].embedding)
        knowledge_base_embeddings = np.array(knowledge_base_embeddings)
    except Exception as e:
        raise RuntimeError(f"Error loading knowledge base: {e}")

def retrieve_relevant_passages(query, top_n=1):
    """
    Retrieves the most relevant passages based on the query using cosine similarity.

    Args:
        query (str): The user's query.
        top_n (int): Number of top relevant passages to retrieve.

    Returns:
        list: Relevant passages from the knowledge base.
    """
    try:
        # Generate embedding for the query
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT_ID,
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)

        # Compute cosine similarity between query and knowledge base embeddings
        scores = np.dot(knowledge_base_embeddings, query_embedding) / (
            np.linalg.norm(knowledge_base_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Debugging: Log similarity scores and questions for analysis
        print(f"Query: {query}")
        for idx, score in enumerate(scores):
            print(f"Score: {score:.4f}, Question: {knowledge_base[idx]['question']}")

        # Filter passages based on a similarity threshold
        threshold = 0.8  # Adjust as needed
        top_indices = [i for i, score in enumerate(scores) if score > threshold]

        # If no passages meet the threshold, return an empty list
        if not top_indices:
            print("No relevant passages meet the threshold.")
            return []

        # Sort indices by descending scores to pick the most relevant passages
        top_indices = sorted(top_indices, key=lambda i: scores[i], reverse=True)[:top_n]

        # Retrieve the most relevant passages
        relevant_passages = [knowledge_base[i] for i in top_indices]

        # Debugging: Log the selected passages
        print("Selected Relevant Passages:")
        for passage in relevant_passages:
            print(f"Q: {passage['question']}, A: {passage['answer']}")

        return relevant_passages
    except Exception as e:
        raise RuntimeError(f"Error retrieving relevant passages: {e}")



def generate_response(query, context):
    """
    Generates a response using the Azure OpenAI GPT model.

    Args:
        query (str): The user's query.
        context (str): Context to include in the model's prompt.

    Returns:
        str: Generated response.
    """
    try:
        completion = client.chat.completions.create(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            messages=[
                {"role": "system", "content": "You are a helpful virtual assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

def synthesize_final_output(query, response, relevant_passages):
    """
    Combines relevant passages and the generated response into a user-friendly output.

    Args:
        query (str): The user's query.
        response (str): Initial response generated by the model.
        relevant_passages (list): Relevant passages from the knowledge base.

    Returns:
        str: Synthesized output.
    """
    try:
        if not relevant_passages:
            combined_context = (
                f"The user asked: {query}\n"
                f"No relevant FAQs were found in the database."
                f"Here is the initial response from the assistant:\n"
                f"{response}\n"
                f"Please refine this into a user-friendly final output."
            )
        else:
            combined_context = (
                f"The user asked: {query}\n"
                f"Here's what we found from the knowledge base:\n"
                f"{''.join([f'Q: {entry['question']} A: {entry['answer']}\n' for entry in relevant_passages])}"
                f"And here's the initial response from the assistant:\n"
                f"{response}\n"
                f"Please synthesize this into a final, user-friendly answer."
            )
        completion = client.chat.completions.create(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled in summarizing and synthesizing information."},
                {"role": "user", "content": combined_context}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error synthesizing final output: {e}")

@app.route('/ask', methods=['POST'])
@limiter.limit("10/minute")  # Protect the endpoint with rate limiting
def ask():
    """
    Endpoint to handle user queries. Retrieves relevant passages, generates a response,
    and synthesizes the final output.
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query not provided"}), 400

        query = data['query']

        # Step 1: Retrieve relevant passages
        relevant_passages = retrieve_relevant_passages(query)

        if not relevant_passages:
            context = "No relevant FAQs were found in the database."
            initial_response = "I'm sorry, I couldn't find any relevant FAQs in the database."
        else:
            context = "\n".join([f"Q: {entry['question']} A: {entry['answer']}" for entry in relevant_passages])
            initial_response = generate_response(query, context)

        # Step 3: Synthesize final output
        final_output = synthesize_final_output(query, initial_response, relevant_passages)

        # Return all components in the JSON response
        return jsonify({
            "query": query,
            "initial_response": initial_response,
            "relevant_passages": relevant_passages,
            "final_output": final_output
        })
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route('/')
def home():
    """
    Renders the HTML home page for the application.
    """
    return render_template('index.html')

if __name__ == "__main__":
    try:
        # Load the knowledge base and precompute embeddings
        load_knowledge_base("bank_faq_questions_answers.csv")
        # Run the Flask app
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start the application: {e}")
