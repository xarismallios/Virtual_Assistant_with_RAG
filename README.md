<img width="764" alt="image" src="https://github.com/user-attachments/assets/674ecd9d-0a20-4eda-b16e-886f1fc4b104" /># 🤖 Virtual_Assistant_with_RAG 🤖 #

A virtual assistant using RAG enhancement. We are using bank's FAQ for frequent clients' questions, which is mapped out with corresponding answers, to help LLM answer user's questions regarding bank features.



Below you will find the relevant architecture

<img width="1000" height="400" alt="image" src= "https://github.com/xarismallios/Virtual_Assistant_with_RAG/blob/main/architecture/ai_llm_rag-5.drawio.svg" />

## Features

- 🌟 **Documents for RAG retrival**
- 🚀 **Hybrid response using LLM**

Create a `.env` file:

```bash
JWT_SECRET=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT=""
COMPLETION_DEPLOYMENT_ID=""
EMBEDDING_DEPLOYMENT_ID=""
```

Run the app using Gunicorn:
```
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

<img width="764" alt="image" src="https://github.com/user-attachments/assets/77a1f8e4-8b3f-4182-b5ee-0b1355db8a52" />


## Acknowledgments

- [Flask](https://flask.palletsprojects.com/): A lightweight and flexible web framework.
- [Gunicorn](https://gunicorn.org/): A robust WSGI server ideal for production deployment.
- [Nginx](https://nginx.org/): A powerful reverse proxy server for handling HTTP requests efficiently.

