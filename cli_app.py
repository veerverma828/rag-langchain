import os
import re
import sys
import time
import random
import shutil
import datetime

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_classic.indexes import SQLRecordManager, index


# ---------------------------------------------------------------------------
# Setup (same models/objects as app.py, just built once - no caching needed
# since this script only runs top-to-bottom a single time, not on every click)
# ---------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen2.5-coder:7b")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})

record_manager = SQLRecordManager(namespace="rag_chain/docs", db_url="sqlite:///record_manager.db")
record_manager.create_schema()

agent_llm_base = ChatOllama(model="llama3.2:3b", temperature=0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
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


agent_tools_by_name = {calculate.name: calculate, get_current_time.name: get_current_time}

prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""")

generation_chain = prompt | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Document syncing
# ---------------------------------------------------------------------------
def get_folder_size_mb(path):
    if not os.path.isdir(path):
        return 0.0
    total_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, dirs, files in os.walk(path)
        for f in files
    )
    return total_bytes / (1024 * 1024)


def _with_embed_progress(chunks, batch_size):
    """Yields chunks one at a time while printing progress every batch_size
    chunks - index() consumes roughly one batch at a time, so this lines up
    closely with how many chunks have actually been embedded so far."""
    total = len(chunks)
    for i, chunk in enumerate(chunks, start=1):
        yield chunk
        if i % batch_size == 0 or i == total:
            print(f"Embedded {i} / {total} chunks", end="\r", flush=True)


def sync_store(folder):
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
    for i, filepath in enumerate(filepaths):
        print(f"Processing {os.path.basename(filepath)} ({i + 1}/{len(filepaths)})", end="\r", flush=True)
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

    print(" " * 80, end="\r")  # clear the progress line

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    batch_size = 100
    print(f"Embedding and indexing {len(chunks)} chunk(s) - this can take a while, please wait...")
    result = index(
        _with_embed_progress(chunks, batch_size),
        record_manager,
        vector_store,
        batch_size=batch_size,
        cleanup="scoped_full",
        source_id_key="source",
        key_encoder="blake2b",
    )
    print()  # move past the progress line
    result["num_deleted"] += num_removed_files
    return result, skipped


def cmd_sync(folder):
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return False
    result, skipped = sync_store(folder)
    print(f"Added: {result['num_added']}, Updated: {result['num_updated']}, "
          f"Skipped (unchanged): {result['num_skipped']}, Deleted: {result['num_deleted']}")
    if skipped:
        print(f"{len(skipped)} file(s) skipped:")
        for msg in skipped:
            print(f"  - {msg}")
    return True


def cmd_files():
    stored = vector_store.get()
    sources = sorted(set(m["source"] for m in stored["metadatas"])) if stored["metadatas"] else []
    if not sources:
        print("No documents indexed yet.")
        return
    print(f"Indexed: {len(sources)} file(s)")
    for f in sources:
        print(f"  - {f}")


def cmd_open(filename):
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return
    os.startfile(filename)


def cmd_storage():
    chroma_size = get_folder_size_mb("chroma_db")
    uploads_size = get_folder_size_mb("uploaded_docs")
    print(f"Vector database: {chroma_size:.2f} MB")
    print(f"Uploaded files:  {uploads_size:.2f} MB")
    print(f"Total:           {chroma_size + uploads_size:.2f} MB")


def cmd_clear():
    confirm = input("This deletes the vector database, record manager, and uploaded files. "
                     "This cannot be undone. Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Cancelled.")
        return
    if os.path.isdir("chroma_db"):
        shutil.rmtree("chroma_db")
    if os.path.isdir("uploaded_docs"):
        shutil.rmtree("uploaded_docs")
    if os.path.isfile("record_manager.db"):
        os.remove("record_manager.db")
    print("All local data deleted. Restart the app to reinitialize a fresh store.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------
def _clean_suggestion_line(line):
    line = line.strip()
    return re.sub(r"^(\d+[\.\)]\s*|[-•]\s*)", "", line).strip()


def generate_suggested_questions():
    all_ids = vector_store.get(include=[])["ids"]
    if not all_ids:
        return []

    sampled_ids = random.sample(all_ids, min(5, len(all_ids)))
    stored = vector_store.get(ids=sampled_ids)
    sample_text = "\n\n".join(stored["documents"])[:2000]

    collected = []
    buffer = ""
    for chunk in llm.stream(
        f"Based on this content, suggest exactly 2 short example questions a user might ask "
        f"(one per line, no numbering, no quotes):\n\n{sample_text}"
    ):
        buffer += chunk.content
        while "\n" in buffer and len(collected) < 2:
            line, buffer = buffer.split("\n", 1)
            line = _clean_suggestion_line(line)
            if line:
                collected.append(line)
        if len(collected) >= 2:
            break

    if len(collected) < 2:
        line = _clean_suggestion_line(buffer)
        if line:
            collected.append(line)

    return collected[:2]


# ---------------------------------------------------------------------------
# Answering logic
# ---------------------------------------------------------------------------
def stream_print(chain_or_llm, arg):
    """Streams tokens to the terminal live, like st.write_stream did, and
    returns the full text once done."""
    full = ""
    for chunk in chain_or_llm.stream(arg):
        text = chunk if isinstance(chunk, str) else chunk.content
        print(text, end="", flush=True)
        full += text
    print()
    return full


def run_document_search(question, query):
    scored = vector_store.similarity_search_with_score(query, k=2)
    retrieved_docs = [doc for doc, score in scored]
    context = format_docs(retrieved_docs)
    answer = stream_print(generation_chain, {"context": context, "question": question})

    if "don't know" not in answer.lower():
        print("\n" + "-" * 60)
        print("Sources:")
        for doc, score in scored:
            preview = doc.page_content[:250].replace("\n", " ")
            print(f"\n  [{score:.4f}] {doc.metadata['source']}")
            print(f"  \"{preview}...\"")
        print("-" * 60)
    return answer


def run_tool(tool_name, args, question):
    tool_fn = agent_tools_by_name[tool_name]
    result = tool_fn.invoke(args)
    if result.startswith("TOOL ERROR"):
        print(f"Tool failed: {result}")
        return f"Tool failed: {result}"
    phrase_prompt = (
        f"Question: {question}\nTool result: {result}\n\n"
        f"Answer the question naturally using this result, in one short sentence."
    )
    return stream_print(llm, phrase_prompt)


MODE_DOCS = "rag"
MODE_TOOL = "tool"
MODE_DIRECT = "ai"
MODES = [MODE_DOCS, MODE_TOOL, MODE_DIRECT]

tool_llm = agent_llm_base.bind_tools([calculate, get_current_time])


def handle_question(question, mode, history):
    start = time.time()

    if mode == MODE_DIRECT:
        answer = stream_print(llm, question)

    elif mode == MODE_DOCS:
        answer = run_document_search(question, question)

    else:  # MODE_TOOL
        decision = tool_llm.invoke([{"role": "user", "content": question}])
        if decision.tool_calls:
            call = decision.tool_calls[0]
            answer = run_tool(call["name"], call["args"], question)
        else:
            print("Couldn't tell whether that needs the calculator or the clock.")
            answer = "Couldn't tell whether that needs the calculator or the clock."

    elapsed = time.time() - start
    print(f"\n(answered in {elapsed:.2f}s)\n")
    history.append((question, answer))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
HELP_TEXT = """
Commands:
  :mode <rag|tool|ai>                  switch answering mode (current: {mode})
  :sync <folder>                       load/re-index documents from a folder (default: docs)
  :files                               list indexed files
  :open <filename>                     open an indexed file in its default app
  :suggest                             show 2 example questions based on your docs
  :storage                             show disk usage of the vector db / uploads
  :export                              save the conversation to conversation.txt
  :clear                               delete all local data (irreversible)
  reset                                start over fresh, like re-running the file
  :help                                show this message
  :exit                                quit

Anything else you type is treated as a question to ask.
"""


def run_command(cmd, arg, mode, history, folder):
    """Executes one ':command'. Returns the (possibly updated) mode and
    folder, and an action - None to keep going, 'exit' to quit, or 'reset'
    to start over."""
    if cmd in ("exit", "quit"):
        return mode, folder, "exit"
    elif cmd == "reset":
        return mode, folder, "reset"
    elif cmd == "help":
        print(HELP_TEXT.format(mode=mode))
    elif cmd == "mode":
        if arg in MODES:
            mode = arg
            print(f"Mode set to: {mode}")
        else:
            print(f"Unknown mode '{arg}'. Choose from: {', '.join(MODES)}")
    elif cmd == "sync":
        target = arg or "docs"
        if cmd_sync(target):
            folder = target
    elif cmd == "files":
        cmd_files()
    elif cmd == "open":
        cmd_open(arg)
    elif cmd == "suggest":
        suggestions = generate_suggested_questions()
        if suggestions:
            print("Try asking:")
            for q in suggestions:
                print(f"  - {q}")
            print()
        else:
            print("No documents indexed yet.")
    elif cmd == "storage":
        cmd_storage()
    elif cmd == "export":
        transcript = "\n\n".join(f"You: {q}\nAI: {a}" for q, a in history)
        with open("conversation.txt", "w", encoding="utf-8") as f:
            f.write(transcript)
        print("Saved to conversation.txt")
    elif cmd == "clear":
        cmd_clear()
    else:
        print(f"Unknown command: {cmd}. Type :help for the list.")
    return mode, folder, None


def run_session():
    """Runs one full session from the folder prompt through the chat loop.
    Returns 'reset' to start over fresh (like re-running the file), or
    'exit' to quit the program entirely."""
    print("*" * 60)
    print("Press h to show all commands.")
    mode = MODE_DOCS
    history = []
    folder = None

    while True:
        typed = input("Folder to sync (e.g. 'docs'), blank to skip: ").strip()
        if typed.lower() == "h":
            print(HELP_TEXT.format(mode=mode))
            continue
        if typed.lower() == "reset":
            return "reset"
        if typed.startswith(":"):
            parts = typed[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            mode, folder, action = run_command(cmd, arg, mode, history, folder)
            if action:
                return action
            if mode == MODE_TOOL:
                print("Tool mode doesn't need documents - skipping folder sync.")
                break
            continue
        typed_folder = typed
        break
    if mode == MODE_TOOL:
        pass
    elif typed_folder and typed_folder.lower() != "skip":
        if cmd_sync(typed_folder):
            folder = typed_folder
    elif not typed_folder:
        print("Skipping sync - using whatever was already indexed before.")

    while True:
        folder_label = folder or "-"
        try:
            line = input(f"[{mode}] [{folder_label}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return "exit"

        if not line:
            continue

        if line.lower() == "h":
            print(HELP_TEXT.format(mode=mode))
            continue

        if line.lower() == "reset":
            return "reset"

        if line.startswith(":"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            mode, folder, action = run_command(cmd, arg, mode, history, folder)
            if action:
                return action
            continue

        print("AI: ", end="")
        handle_question(line, mode, history)


def main():
    while run_session() == "reset":
        print("\nResetting...\n")


if __name__ == "__main__":
    main()
