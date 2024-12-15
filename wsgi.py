from main import app 


if __name__ == "__main__":
    try:
        # Load the knowledge base and precompute embeddings
        load_knowledge_base("bank_faq_questions_answers.csv")
        # Run the Flask app
        app.run(host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Failed to start the application: {e}")