# 🤖 AI Virtual Assistant using RAG 🤖 

[![Python Version](https://img.shields.io/badge/python-3.12.7%2B-blue)](https://www.python.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/username/project)](LICENSE)

👟 **NikeBot** provides a cutting-edge, production-ready solution for managing sneaker drops and restocks, so you never miss your chance to cop your favorite kicks! 🚀


An AI Virtual Assistant using RAG. We are an e-commerce Brand selling washing machines. Our customers oftern trying to find info about our proudcts and classic LLMs and chatGPT doesn't have relative and up to date info. On the other hand, We have relevant documents, including machines manuals from the manufacturer, reviews from customers or even past q&a topics we have gathered through out customer support mail system. So we are using RAG to retreive those documents and enhance our knowledge base with more relative content in users queries. 


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

