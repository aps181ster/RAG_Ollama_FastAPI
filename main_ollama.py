# main_ollama.py

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import faiss
import numpy as np
from preprocess import preprocess_documents  # Import the preprocess function
from ollama import Client  # Import the Ollama Client for embedding generation
import os

# Initialize Ollama Client for generating embeddings
client = Client()

# Load FAISS index (assuming the FAISS index is created with the embeddings from preprocess.py)
index_path = 'faiss_index'
faiss_index = faiss.read_index(index_path)

# Initialize Ollama API URL and models
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Local URL for Ollama API
OLLAMA_GENERATION_MODEL = "qwen3:0.6b"  # Model for generating responses
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:v1.5"  # Model for generating embeddings

# Path to the TXT files directory
txt_directory = 'documents/'

# List all TXT files in the directory
txt_files = [os.path.join(txt_directory, f) for f in os.listdir(txt_directory) if f.endswith('.txt')]

# Preprocess documents to get chunks and embeddings
document_chunks, embeddings = preprocess_documents(txt_files)

# Initialize FastAPI app
app = FastAPI()

class StoryRequest(BaseModel):
    prompt: str
    max_tokens: int = 500

# Function to retrieve documents based on query
def retrieve_documents(query: str, top_k: int = 3):
    # Use Ollama Client to generate embedding for the query using the embedding model
    response = client.embed(model=OLLAMA_EMBEDDING_MODEL, input=query)
    
    # Log the full response for debugging
    print("Full response from Ollama API:", response)

    embedding_key = 'embeddings'
    if embedding_key not in response:
        return {"error": "Embedding not found in the response from Ollama API"}
    
    # Extract the embedding from the response
    query_embedding = response[embedding_key][0]

    # Ensure that the embedding is a valid numpy array
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding).astype('float32')

    # Reshape query_embedding to 2D (1, d) for FAISS
    query_embedding = query_embedding.reshape(1, -1)

    # Perform vector search in FAISS
    distances, retrieved_docs = faiss_index.search(query_embedding, top_k)
    retrieved_docs = retrieved_docs.flatten()  # Ensure indices are a 1D array of ints

    # Log retrieved documents to see what is being fetched
    print("Retrieved documents:", [document_chunks[i] for i in retrieved_docs])

    if not len(retrieved_docs):
        return {"error": "No relevant documents found in the index."}

    return [document_chunks[i] for i in retrieved_docs]

@app.post("/generate-story")
def generate_story(request: StoryRequest):
    # Step 1: Retrieve relevant documents from FAISS index
    retrieved_docs = retrieve_documents(request.prompt, top_k=3)
    
    # If no relevant documents found
    if isinstance(retrieved_docs, dict) and 'error' in retrieved_docs:
        return retrieved_docs
    
    # Step 2: Combine retrieved documents with the prompt
    context = "\n\n".join(retrieved_docs)
    augmented_prompt = f"Here is some information from documents:\n\n{context}\n\nBased on the context, answer the following question: {request.prompt}"
    
    # Step 3: Call Ollama API with the augmented prompt using the generation model
    payload = {
        "model": OLLAMA_GENERATION_MODEL,
        "prompt": augmented_prompt,
        "options": {
            "num_predict": request.max_tokens
        },
        "stream": False
    }

    response = requests.post(OLLAMA_API_URL, json=payload)

    if response.status_code != 200:
        return {"error": response.text}

    data = response.json()
    response_text = data.get("response", "")

    return {"response": response_text}
