from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import glob

file_paths = glob.glob("docs/**/*.txt", recursive=True) + glob.glob("docs/**/*.pdf", recursive=True)

loader = UnstructuredLoader(file_path=file_paths, strategy="fast", chunking_strategy="by_title")
chunks = loader.load()
print(f"Loaded {len(chunks)} chunks")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)
print(f"Stored {len(chunks)} chunks in Chroma at ./chroma_db")