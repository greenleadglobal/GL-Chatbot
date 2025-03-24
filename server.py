from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

app = FastAPI()

index = None

@app.on_event("startup")
def load_vectorstore():
    global load_vectorstore

    try:
        embedding = HuggingfaceBgeEmbeddings(model_name = 'sentence-transformers/all-miniLM-L6-v2')

        vectorstore = FAISS.load_local('vectorstore')
        print('FAISS vectorstore loaded')

    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise HTTPException(status_code=500, detail="Error loading vectorstore")
    

@app.get('/')
def home():
    return {'message': 'Vectorstore API is running'}


