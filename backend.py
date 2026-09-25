#for llm
import chromadb
from chromadb.config import Settings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
#for tool mode
import datetime
from langchain_core.tools import tool
from ddgs import DDGS
#for syncing documents
import os
import json
import threading
import time
import shutil
import urllib.request
import urllib.error
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.indexes import SQLRecordManager, index
#Main backend
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
#for serving files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse


#main llm setup
DEFAULT_MODEL = "qwen2.5-coder:7b"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Cross-encoder reranker: scores each (question, chunk) pair together for
# actual relevance, unlike embedding similarity which compares them
# independently. Loaded once at startup, not via Ollama - Ollama has no
# native reranking endpoint, so this uses the standard local Python approach.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# allow_reset=True enables client.reset() - the documented, official chromadb
# API for wiping a database - used by the /clear endpoint below.
chroma_client = chromadb.PersistentClient(path="chroma_db", settings=Settings(allow_reset=True))
# cosine matches how nomic-embed-text (like most text embedding models) was
# actually trained, unlike Chroma's raw-distance default ("l2") - this only
# takes effect on a freshly created collection, so an existing chroma_db
# needs a /clear + re-sync to actually pick it up.
vector_store = Chroma(client=chroma_client, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})

record_manager = SQLRecordManager(namespace="rag_chain/docs", db_url="sqlite:///record_manager.db")
record_manager.create_schema()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the context below. If the answer isn't in the "
               "context, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# Short-term conversation memory: this app is single-user/local, so one
# global in-memory list is enough - no per-session tracking needed. It's
# capped so a long-running session doesn't grow the prompt (and therefore
# the context window sent to a CPU-bound model) without bound.
conversation_history = []
MAX_HISTORY_MESSAGES = 20

def record_turn(question, answer):
    conversation_history.append(HumanMessage(content=question))
    conversation_history.append(AIMessage(content=answer))
    while len(conversation_history) > MAX_HISTORY_MESSAGES:
        conversation_history.pop(0)

@tool
def calculate(expression: str) -> str:
    """Evaluate a NUMERIC math expression using only digits and +-*/(). Only
    use this for arithmetic questions. Do NOT use it for facts, names, or
    anything that isn't a plain math calculation."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return (f"TOOL ERROR: '{expression}' is not a valid math expression "
                f"(contains non-numeric characters). This tool only accepts "
                f"digits and + - * / ( ). Do not retry with a different, "
                f"non-numeric expression - this question is not a math problem.")
    try:
        return str(eval(expression))
    except Exception as e:
        return f"TOOL ERROR: could not evaluate '{expression}': {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def web_search(query: str) -> str:
    """Search the web for current information, recent events, news, or facts
    that would not be in the user's own documents. Do NOT use it for math
    or for the current date/time."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"TOOL ERROR: web search failed: {e}"
    if not results:
        return "TOOL ERROR: no results found for that search."
    return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results)

tools = [calculate, get_current_time, web_search]
agent_tools_by_name = {t.name: t for t in tools}

# The generation model is now chosen per-request from whatever the user picks
# in the UI, instead of being fixed at startup. ChatOllama clients are cheap
# but not free to construct, so cache one per model name instead of building
# a new one on every single request.
_llm_cache = {}
_tool_llm_cache = {}

def get_llm(model_name):
    if model_name not in _llm_cache:
        _llm_cache[model_name] = ChatOllama(model=model_name)
    return _llm_cache[model_name]

def get_tool_llm(model_name):
    if model_name not in _tool_llm_cache:
        _tool_llm_cache[model_name] = ChatOllama(model=model_name, temperature=0).bind_tools(tools)
    return _tool_llm_cache[model_name]


def sync_store_stream(folder):
    """Generator that yields newline-delimited JSON progress updates while
    syncing a folder, ending with a 'done' message containing the result.
    Runs the actual embedding/indexing in a background thread so this
    generator can keep reporting progress while it's in flight."""
    filepaths = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            filepaths.append(os.path.join(root, filename))

    # cleanup="scoped_full" (below) only cleans up sources it actually sees
    # during this sync - a file deleted from disk is never walked, so its old
    # chunks would otherwise be silently orphaned forever. Detect that case
    # ourselves and remove just those chunks, scoped only to this folder.
    stored = vector_store.get(include=["metadatas"])
    existing_sources = {m["source"] for m in stored["metadatas"] if m["source"].startswith(folder)}
    current_sources = set(filepaths)
    deleted_sources = existing_sources - current_sources

    num_removed_files = 0
    if deleted_sources:
        to_delete = vector_store.get(where={"source": {"$in": list(deleted_sources)}}, include=[])
        if to_delete["ids"]:
            vector_store.delete(ids=to_delete["ids"])
        keys = record_manager.list_keys(group_ids=list(deleted_sources))
        if keys:
            record_manager.delete_keys(keys)
        num_removed_files = len(deleted_sources)

    all_docs = []
    skipped = []
    for filepath in filepaths:
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext in (".txt", ".md"):
                docs = TextLoader(filepath, encoding="utf-8").load()
            elif ext == ".pdf":
                docs = PyPDFLoader(filepath).load()
            elif ext == ".docx":
                docs = Docx2txtLoader(filepath).load()
            else:
                skipped.append(f"{filepath}: unsupported file type")
                continue
        except Exception as e:
            skipped.append(f"{filepath}: {e}")
            continue
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    total = len(chunks)

    yield json.dumps({"type": "progress", "done": 0, "total": total}) + "\n"

    if total == 0:
        result = index([], record_manager, vector_store, cleanup="scoped_full", source_id_key="source", key_encoder="blake2b")
        result["num_deleted"] += num_removed_files
        yield json.dumps({"type": "done", "result": result, "skipped": skipped}) + "\n"
        return

    progress = {"done": 0}
    outcome = {}
    batch_size = 100

    def counting_chunks():
        for i, chunk in enumerate(chunks, start=1):
            yield chunk
            progress["done"] = i

    def run_index():
        outcome["result"] = index(
            counting_chunks(), record_manager, vector_store,
            batch_size=batch_size, cleanup="scoped_full",
            source_id_key="source", key_encoder="blake2b",
        )

    thread = threading.Thread(target=run_index)
    thread.start()

    last_reported = -1
    while thread.is_alive():
        if progress["done"] != last_reported:
            yield json.dumps({"type": "progress", "done": progress["done"], "total": total}) + "\n"
            last_reported = progress["done"]
        time.sleep(0.3)
    thread.join()

    yield json.dumps({"type": "progress", "done": total, "total": total}) + "\n"
    outcome["result"]["num_deleted"] += num_removed_files
    yield json.dumps({"type": "done", "result": outcome["result"], "skipped": skipped}) + "\n"

#routings-
app = FastAPI()

@app.get("/")
def read_index():
    return FileResponse("static/index.html")
#for static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Question(BaseModel):
    question: str
    mode: str = "rag"
    folder: str = ""
    model: str = DEFAULT_MODEL

@app.get("/models")
def list_models():
    # Query Ollama's own local API directly (same thing "ollama list" does)
    # instead of hardcoding a model list, so newly pulled models show up
    # without a code change.
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        names = [m["name"] for m in data.get("models", [])]
        return {"models": names}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"models": [], "error": f"Could not reach Ollama: {e}"}

def stream_tokens(chunks, sources, question=None):
    """Wrap a LangChain .stream() iterator into NDJSON token lines, ending
    with a 'done' line carrying the sources (chat models yield AIMessageChunk
    objects with .content; a chain ending in StrOutputParser yields plain
    strings - handle both). If `question` is given, the full accumulated
    answer is recorded into conversation_history once streaming finishes."""
    full_text = ""
    for chunk in chunks:
        token = chunk.content if hasattr(chunk, "content") else chunk
        if token:
            full_text += token
            yield json.dumps({"type": "token", "content": token}) + "\n"
    if question is not None:
        record_turn(question, full_text)
    yield json.dumps({"type": "done", "sources": sources}) + "\n"


def status(text):
    return json.dumps({"type": "status", "content": text}) + "\n"


@app.post("/ask")
def ask(q: Question):
    llm = get_llm(q.model)

    if q.mode == "ai":
        def ai_stream():
            messages = conversation_history + [HumanMessage(content=q.question)]
            yield from stream_tokens(llm.stream(messages), [], question=q.question)
        return StreamingResponse(ai_stream(), media_type="application/x-ndjson")

    if q.mode == "tool":
        def tool_stream():
            tool_llm = get_tool_llm(q.model)
            yield status("Deciding which tool to use...")
            decision = tool_llm.invoke([{"role": "user", "content": q.question}])
            if not decision.tool_calls:
                yield json.dumps({"type": "token", "content": "Couldn't tell whether that needs the calculator or the clock."}) + "\n"
                yield json.dumps({"type": "done", "sources": []}) + "\n"
                return

            call = decision.tool_calls[0]
            tool_fn = agent_tools_by_name[call["name"]]
            yield status(f"Running {call['name']}...")
            result = tool_fn.invoke(call["args"])
            if result.startswith("TOOL ERROR"):
                yield json.dumps({"type": "token", "content": f"Tool failed: {result}"}) + "\n"
                yield json.dumps({"type": "done", "sources": []}) + "\n"
                return

            phrase_prompt = (
                f"Question: {q.question}\nTool result: {result}\n\n"
                f"Answer the question naturally using only this result, in 1-3 short sentences."
            )
            yield status("Answering...")
            messages = conversation_history + [HumanMessage(content=phrase_prompt)]
            yield from stream_tokens(llm.stream(messages), [], question=q.question)
        return StreamingResponse(tool_stream(), media_type="application/x-ndjson")

    # rag mode
    def rag_stream():
        yield status("Searching documents...")
        # Chroma can't filter by "starts with" natively, so fetch extra
        # candidates and filter down to the selected folder in Python
        candidates = vector_store.similarity_search_with_score(q.question, k=10)
        if q.folder:
            candidates = [(doc, score) for doc, score in candidates if doc.metadata["source"].startswith(q.folder)]

        # Rerank: embedding distance is a cheap first pass, but comparing the
        # question and each chunk independently misses a lot of nuance. The
        # cross-encoder scores them together instead, so the final top-2
        # picked here is by actual relevance, not just vector proximity -
        # note this score is "higher is better", the opposite of the
        # distance scores above.
        docs_only = [doc for doc, _ in candidates]
        if docs_only:
            yield status("Reranking results...")
            pairs = [(q.question, doc.page_content) for doc in docs_only]
            rerank_scores = reranker.predict(pairs)
            scored = sorted(zip(docs_only, rerank_scores), key=lambda x: x[1], reverse=True)[:2]
        else:
            scored = []

        docs = [doc for doc, score in scored]
        context = "\n\n".join(doc.page_content for doc in docs)

        sources = [
            {"source": doc.metadata["source"], "score": float(score), "preview": doc.page_content[:200]}
            for doc, score in scored
        ]
        yield status("Generating answer...")
        generation_chain = prompt | llm | StrOutputParser()
        yield from stream_tokens(
            generation_chain.stream({"context": context, "question": q.question, "history": conversation_history}),
            sources, question=q.question
        )
    return StreamingResponse(rag_stream(), media_type="application/x-ndjson")


class SyncRequest(BaseModel):
    folder: str

@app.post("/sync")
def sync(req: SyncRequest):
    if not os.path.isdir(req.folder):
        def error_stream():
            yield json.dumps({"type": "error", "error": f"Folder not found: {req.folder}"}) + "\n"
        return StreamingResponse(error_stream(), media_type="application/x-ndjson")
    return StreamingResponse(sync_store_stream(req.folder), media_type="application/x-ndjson")


@app.get("/files")
def list_files(folder: str = ""):
    stored = vector_store.get()
    sources = sorted(set(m["source"] for m in stored["metadatas"])) if stored["metadatas"] else []
    if folder:
        sources = [s for s in sources if s.startswith(folder)]
    return {"files": sources}


class OpenRequest(BaseModel):
    path: str

@app.post("/open")
def open_file(req: OpenRequest):
    if not os.path.isfile(req.path):
        return {"error": f"File not found: {req.path}"}
    os.startfile(req.path)
    return {"opened": req.path}


UPLOAD_FOLDER = "uploaded_docs"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    return StreamingResponse(sync_store_stream(UPLOAD_FOLDER), media_type="application/x-ndjson")


def get_folder_size_mb(path):
    if not os.path.isdir(path):
        return 0.0
    total_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, dirs, files in os.walk(path)
        for f in files
    )
    return total_bytes / (1024 * 1024)


@app.get("/storage")
def storage():
    # Report the size of the actual data stored (embeddings + document
    # text), not the raw .sqlite3 file size on disk - a database file always
    # has some fixed structural overhead even when empty, which made "0
    # documents" confusingly show as a non-zero size before.
    stored = vector_store.get(include=["documents", "embeddings"])
    num_chunks = len(stored["ids"])
    if num_chunks > 0 and stored["embeddings"] is not None and len(stored["embeddings"]) > 0:
        dimension = len(stored["embeddings"][0])
        vectors_bytes = num_chunks * dimension * 4  # float32 = 4 bytes each
    else:
        vectors_bytes = 0
    text_bytes = sum(len(doc.encode("utf-8")) for doc in stored["documents"]) if stored["documents"] else 0
    chroma_mb = (vectors_bytes + text_bytes) / (1024 * 1024)

    uploads_mb = get_folder_size_mb(UPLOAD_FOLDER)
    return {"chroma_mb": chroma_mb, "uploads_mb": uploads_mb, "total_mb": chroma_mb + uploads_mb}


@app.post("/clear")
def clear():
    # the server keeps chroma_db and record_manager.db open the whole time
    # it's running, so we wipe their contents through the libraries' own
    # documented APIs, using the single shared client/connection the app
    # already holds - never a second, separate connection to the same live
    # file, which is what caused real data corruption when this was tried.
    global vector_store

    # reset() is chromadb's own official, documented API for fully wiping a
    # database (requires allow_reset=True, set on chroma_client above).
    chroma_client.reset()
    vector_store = Chroma(client=chroma_client, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})

    # reset() clears the catalog but, on this chromadb version, doesn't
    # reliably delete every collection's on-disk folder (data_level0.bin
    # etc.) - clean up anything left behind that no longer matches a real,
    # currently-registered collection. This only touches folders that are
    # no longer referenced by any live collection, so it's safe.
    valid_ids = {str(c.id) for c in chroma_client.list_collections()}
    for entry in os.listdir("chroma_db"):
        entry_path = os.path.join("chroma_db", entry)
        if os.path.isdir(entry_path) and entry not in valid_ids:
            shutil.rmtree(entry_path)

    keys = record_manager.list_keys()
    if keys:
        record_manager.delete_keys(keys)

    if os.path.isdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)

    return {"cleared": True}