import chromadb
import os

db_dir = r"c:\Users\Mahir sakhuja\New folder (4)\Day_7_Course_Material\Samsung_RAG_Deployment\chroma_db"
if not os.path.exists(db_dir):
    print("Database directory not found!")
    exit(1)

client = chromadb.PersistentClient(path=db_dir)
collections = client.list_collections()
print(f"Collections: {collections}")

for col in collections:
    print(f"=== Collection: {col.name} ===")
    # Get all items in the collection
    results = col.get()
    print(f"Number of documents: {len(results.get('documents', []))}")
    for i, doc in enumerate(results.get('documents', [])[:10]):
        print(f"Doc {i}: {doc[:300]}")
        print("-" * 40)
    
    # Save all documents to a text file for RAG context
    with open("extracted_samsung_manual.txt", "w", encoding="utf-8") as f:
        for doc in results.get("documents", []):
            f.write(doc + "\n\n========================================\n\n")
    print("Saved all documents to extracted_samsung_manual.txt")
