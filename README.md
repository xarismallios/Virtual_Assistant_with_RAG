# 🤖 Virtual_Assistant_with_RAG 🤖 #

A virtual assistant using RAG enhancement. We are using bank's FAQ for frequent clients' questions, which is mapped out with corresponding answers, to help LLM answer user's questions regarding bank features.



Below you will find the relevant architecture

<img width="1000" height="400" alt="image" src= "https://github.com/xarismallios/Virtual_Assistant_with_RAG/blob/main/ai_llm_rag_Qdrant.drawio.png" />



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

<img width="385" alt="image" src="https://github.com/user-attachments/assets/f567162d-3e61-41a8-a802-aa463ee89678" />


## Acknowledgments

- [Flask](https://flask.palletsprojects.com/): A lightweight and flexible web framework.
- [Gunicorn](https://gunicorn.org/): A robust WSGI server ideal for production deployment.
- [Nginx](https://nginx.org/): A powerful reverse proxy server for handling HTTP requests efficiently.

