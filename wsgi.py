from main import app, load_knowledge_base

# Preload the knowledge base
try:
    load_knowledge_base("bank_faq_questions_answers.csv")
except Exception as e:
    print(f"Error loading knowledge base: {e}")
    raise e

# Gunicorn will look for 'app' here
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
