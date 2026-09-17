from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import glob

# must match the model used to build the store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# model that generates the answer
llm = ChatOllama(model="qwen2.5-coder:7b")

# template with {context} and {question} placeholders
prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""")

# joins retrieved chunks into one text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# NEW: builds (or rebuilds) the vector store from a given folder - lets us point the app
# at any folder, and re-run this to pick up new/changed files (same idea as the manual
# project's build_index() + 'r' reload).
def build_store(folder):
    file_paths = glob.glob(f"{folder}/**/*.txt", recursive=True) + glob.glob(f"{folder}/**/*.pdf", recursive=True)
    loader = UnstructuredLoader(file_path=file_paths, strategy="fast", chunking_strategy="by_title")
    chunks = loader.load()
    print(f"Loaded {len(chunks)} chunks from {folder}")
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma_db")

# NEW: rebuilds the retriever + chain after a (re)build, so they use the fresh vector store
def build_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # k=2 -> top 2 relevant chunks
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# NEW: ask which folder to index, same 's' shortcut as the manual project
docs_folder = input("Enter folder to search (or 's' for docs/): ").strip()
if docs_folder.lower() == "s":
    docs_folder = "docs"

vector_store = build_store(docs_folder)
chain = build_chain(vector_store)

print("RAG ready. Type your question ('r' to rescan folder, 'exit' to quit).")
while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        break
    # NEW: 'r' rebuilds the store from the same folder, picking up new/changed files
    if question.lower() == "r":
        vector_store = build_store(docs_folder)
        chain = build_chain(vector_store)
        print("Reloaded.")
        continue
    answer = chain.invoke(question)  # runs the whole chain
    print(f"\nAI: {answer}")
