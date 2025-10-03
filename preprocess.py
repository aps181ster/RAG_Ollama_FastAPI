# preprocess.py

from ollama import Client  # Import the Client class for generating embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os

# Initialize Ollama model for generating embeddings
client = Client()

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    """
    Extracts text from a TXT file.
    
    Parameters:
    txt_path (str): Path to the TXT file.
    
    Returns:
    str: Extracted text from the TXT file.
    """
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to preprocess documents (split and embed)
def preprocess_documents(txt_files):
    """
    Preprocesses documents by splitting them into chunks and generating embeddings for each chunk.

    Parameters:
    txt_files (list of str): List of TXT file paths to preprocess.

    Returns:
    embeddings (list of list): List of embeddings corresponding to each document chunk.
    texts (list of str): List of text chunks.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    document_chunks = []
    embeddings = []

    # Extract text from each TXT and split into chunks
    for txt_path in txt_files:
        text = extract_text_from_txt(txt_path)
        print(f"Extracted text from {txt_path}:")
        print(text[:200])  # Print first 200 characters of text for verification
        document_chunks.extend(text_splitter.split_text(text))

    print(f"Total chunks generated: {len(document_chunks)}")
    
    # Generate embeddings for each chunk
    for i, chunk in enumerate(document_chunks):
        if len(chunk.strip()) == 0:
            print(f"Skipping empty chunk at index {i}.")
            continue
        
        try:
            # Generate embedding for each chunk using the Ollama Client
            print(f"Generating embedding for chunk {i}: {chunk[:50]}...")  # Log first 50 characters of chunk for debugging
            response = client.embed(model="nomic-embed-text:v1.5", input=chunk)
            
            # Log the full response to inspect any potential issues
            print(f"Embedding response for chunk {i}: {response}")  # Log full response
            
            # Check if embedding is in the response
            if response.get('embeddings') is not None:
                embedding = response['embeddings'][0]  # Assuming the embeddings are in a list
                embeddings.append(embedding)
            else:
                print(f"Warning: No embedding found for chunk {i}: {chunk[:30]}...")  # Print first 30 characters of chunk
                continue
        except Exception as e:
            print(f"Error during embedding generation for chunk {i}: {chunk[:30]}... Error: {e}")

    print(f"Embeddings generated: {len(embeddings)} chunks")
    if embeddings:
        print(f"Sample embedding shape: {np.array(embeddings).shape}")

    return document_chunks, embeddings

# Function to create and save FAISS index
def create_faiss_index(texts, embeddings, index_path='faiss_index'):
    """
    Creates and saves the FAISS index from document embeddings.

    Parameters:
    texts (list of str): List of text chunks.
    embeddings (list of list): List of embeddings corresponding to each document chunk.
    index_path (str): Path to save the FAISS index.

    Returns:
    faiss_index: The created FAISS index.
    """
    # Convert embeddings to numpy array (FAISS works with numpy arrays)
    if not embeddings:
        print("Error: No embeddings to save in FAISS index.")
        return None
    
    embeddings_np = np.array(embeddings).astype('float32')

    # Initialize FAISS index
    dim = embeddings_np.shape[1]  # Dimension of embeddings
    faiss_index = faiss.IndexFlatL2(dim)

    # Add embeddings to the FAISS index
    faiss_index.add(embeddings_np)

    # Save the FAISS index
    faiss.write_index(faiss_index, index_path)
    print(f"FAISS index saved at {index_path}")

    return faiss_index

# Main function to process documents and create FAISS index
def main():
    # Path to the TXT file directory
    txt_directory = 'documents/'

    # List all TXT files in the directory
    txt_files = [os.path.join(txt_directory, f) for f in os.listdir(txt_directory) if f.endswith('.txt')]

    # Step 1: Preprocess documents (split and embed)
    document_chunks, embeddings = preprocess_documents(txt_files)

    # Step 2: Create FAISS index and save
    if embeddings:
        create_faiss_index(document_chunks, embeddings)
    else:
        print("No embeddings generated. FAISS index creation skipped.")

# Run preprocessing
if __name__ == "__main__":
    main()
