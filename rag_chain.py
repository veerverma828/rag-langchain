from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.indexes import SQLRecordManager, index

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

# The vector store itself - created once, reused across runs. Chroma will create an empty
# store here on first run, and load the existing one on every run after that.
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# NEW: the record manager is LangChain's official way to track "what have I already indexed,
# and what does its content look like right now" - it keeps a small SQLite database of
# content hashes per document, so re-syncing can tell exactly what's new/changed/unchanged.
record_manager = SQLRecordManager(
    namespace="rag_chain/docs",
    db_url="sqlite:///record_manager.db",
)
record_manager.create_schema()  # safe to call every run - does nothing if already set up

# UPDATED: loads + chunks the folder, then syncs with the vector store via the record
# manager instead of wiping and rebuilding. cleanup="full" means: anything in the vector
# store that ISN'T in this folder anymore (e.g. a deleted file) gets removed too, since we
# always pass the complete current set of chunks for the folder.
def sync_store(folder):
    txt_docs = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
    pdf_docs = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    all_docs = txt_docs + pdf_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    result = index(
        chunks,
        record_manager,
        vector_store,
        cleanup="full",
        source_id_key="source",
    )
    print(f"Synced {folder}: {result}")

retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # k=2 -> top 2 relevant chunks
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# NEW: ask which folder to sync, same 's' shortcut as before. This always runs, but is now
# cheap - unchanged files get skipped automatically by the record manager, so there's no
# need for a manual 'r' reload command anymore.
docs_folder = input("Enter folder to search (or 's' for docs/): ").strip()
if docs_folder.lower() == "s":
    docs_folder = "docs"

sync_store(docs_folder)

print("RAG ready. Type your question ('exit' to quit).")
while True:
    question = input("\nYou: ")
    if question.lower() == "exit":
        break
    answer = chain.invoke(question)  # runs the whole chain
    print(f"\nAI: {answer}")
