# 🤖 AI Virtual Assistant using RAG 🤖 

[![Python Version](https://img.shields.io/badge/python-3.12.7%2B-blue)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/xarismallios/Virtual_Assistant_with_RAG)](https://github.com/xarismallios/Virtual_Assistant_with_RAG/blob/main/LICENSE)


🤖  **AI Virtual Assistant using RAG** provides a cutting-edge, production-ready solution for answering users questions, based on our latest documentation and user manuals. Users chat with our data !


We are an e-commerce brand specializing in selling washing machines. Our customers often seek information about our products, but traditional LLMs like ChatGPT lack access to accurate and up-to-date details. However, we have access to a wealth of relevant resources, including manufacturer manuals, customer reviews, and a collection of past Q&A topics compiled from our customer support system. To bridge this gap, we utilize a Retrieval-Augmented Generation (RAG) approach to retrieve these documents and enrich our knowledge base, ensuring more relevant and precise responses to user queries.


Below you will find the relevant architecture

<img width="1000" height="400" alt="image" src= "https://github.com/xarismallios/Virtual_Assistant_with_RAG/blob/main/architecture/ai_llm_rag-5.drawio.svg" />

## Features

- 🌟 **Documents for RAG retrival**
- 🚀 **Hybrid response using LLM**


Simply deploy and run the app after deploying a docker image
```
docker-compose up
```

Don't forget to config the `.env` file:

```bash
JWT_SECRET=""
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT=""
COMPLETION_DEPLOYMENT_ID=""
EMBEDDING_DEPLOYMENT_ID=""
```

<img width="764" alt="image" src="https://github.com/user-attachments/assets/77a1f8e4-8b3f-4182-b5ee-0b1355db8a52" />


## Acknowledgments

- [Flask](https://flask.palletsprojects.com/): A lightweight and flexible web framework.
- [Gunicorn](https://gunicorn.org/): A robust WSGI server ideal for production deployment.
- [Nginx](https://nginx.org/): A powerful reverse proxy server for handling HTTP requests efficiently.

