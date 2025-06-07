---
title: paperchase
sdk: gradio
sdk_version: 5.32.1
app_file: src/frontend/gradio_frontend.py
---



---

# paperchase
Recommendation system for background reading

# Langchain/Azure Update (v0.2)

An Azure‑native LangChain application that retrieves computer‑science papers from arXiv, ranks them, and returns concise summaries.

* **Runtime:** Azure Functions (Python 4 model)  
* **LLM + embeddings:** Azure OpenAI (gpt‑4o‑mini, text‑embedding‑3‑small)  
* **Vector store:** Chroma (local) — swap for Azure AI Search later  
* **Infrastructure as Code:** Azure CLI snippets in this README  

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in Azure keys + arXiv email
python -m src.cli "good CS paper for Mandarin‑English RAG"
```

## Deploy to Azure Functions
```bash
az login
az group create -n paperchase-rg -l eastus
az storage account create -g paperchase-rg -n paperchasesa --sku Standard_LRS
func azure functionapp publish paperchase‑func -g paperchase-rg --python
```

## TODO: evaluation/metric explanation go here
```bash
python -m src.eval.run_eval
```
