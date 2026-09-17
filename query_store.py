from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

question = "What CPU is recommended for gaming?"
results = retriever.invoke(question)

print(f"Found {len(results)} relevant chunks\n")
for doc in results:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...\n")