from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="document_embeddings")

# Initialize text-generation model
generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

@app.get("/health")
def health_check():
    """Check API status."""
    return {"status": "OKK"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and index documents (TXT or PDF) by generating embeddings.
    """
    content = await file.read()
    text = content.decode("utf-8")  

    # Compute embedding
    embedding = model.encode([text]).tolist()

    # Check if the embedding already exists (approximate match)
    existing_docs = collection.query(query_embeddings=embedding, n_results=1)
    if existing_docs["ids"] and existing_docs["distances"][0][0] < 0.01:  # Threshold for duplicate detection
        return {"message": "Duplicate document detected. Skipping ingestion."}

    # Assign unique ID
    doc_id = str(len(collection.get()["ids"]))  # Unique ID

    # Store in ChromaDB
    collection.add(ids=[doc_id], embeddings=embedding, metadatas=[{"text": text}])
    
    return {"message": "Document ingested successfully", "doc_id": doc_id}

@app.get("/query")
def query_model(query: str, k: int = 2):
    """
    Retrieve top-k relevant documents and generate a response.
    """
    # Encode query
    query_embedding = model.encode([query]).tolist()
    
    # Retrieve similar documents
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    # Remove duplicate texts
    retrieved_texts = list(set(metadata["text"] for metadata in results["metadatas"][0]))
    
    context = "\n".join(retrieved_texts)

    # Generate response
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer (concise, no redundancy):"
    
    response = generator(prompt, max_new_tokens=100, do_sample=True, pad_token_id=50256)
    
    print("response generated : ",response)
    return {"query": query, "retrieved_documents": retrieved_texts, "generated_response": response[0]["generated_text"]}

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
