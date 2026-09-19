import streamlit as st
import os
import time
import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage
from langchain_classic.indexes import SQLRecordManager, index

st.set_page_config(page_title="Local RAG (LangChain)", page_icon="🔗")

# NEW: lets button text wrap onto multiple lines instead of being cut off with "..."
# (used by the suggested-question chips)
st.markdown("""
<style>
.stButton button, .stButton button p {
    white-space: normal;
    height: auto;
    overflow: visible;
    text-overflow: clip;
    word-wrap: break-word;
}
div[data-testid="stChatInput"] > div:first-child,
div[data-testid="stSelectbox"] [role="group"] {
    min-height: 3.6rem;
    border-radius: 12px !important;
    border: 1px solid rgba(250, 250, 250, 0.15) !important;
    display: flex;
    align-items: center;
}
div[data-testid="stSelectbox"] [role="group"],
div[data-testid="stChatInput"] textarea {
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🔗 Local RAG — LangChain Edition")

# st.cache_resource keeps these objects alive across Streamlit's re-runs, instead of
# recreating the embeddings model / vector store / record manager on every single interaction
# (Streamlit re-runs the whole script top-to-bottom each time you click/type anything).
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def get_llm():
    return ChatOllama(model="qwen2.5-coder:7b")

@st.cache_resource
def get_vector_store():
    return Chroma(persist_directory="chroma_db", embedding_function=get_embeddings())

@st.cache_resource
def get_record_manager():
    rm = SQLRecordManager(namespace="rag_chain/docs", db_url="sqlite:///record_manager.db")
    rm.create_schema()
    return rm

# NEW: a small, separate model dedicated to deciding WHICH tool (if any) a question
# needs. qwen2.5-coder (the main answer-writing model) doesn't reliably emit structured
# tool calls through Ollama - llama3.2:3b does, so it's used only for this routing
# decision. The actual final answer is still written by the bigger qwen2.5-coder model,
# using whatever the chosen tool returned - small model decides, big model writes.
@st.cache_resource
def get_agent_llm():
    return ChatOllama(model="llama3.2:3b", temperature=0)

embeddings = get_embeddings()
llm = get_llm()
vector_store = get_vector_store()
record_manager = get_record_manager()
agent_llm_base = get_agent_llm()

# NEW: tools the agent can choose to call. calculate/get_current_time are executed
# directly through these; search_documents is only defined so the model can see and
# select it - its actual execution reuses similarity_search_with_score() further down,
# so the citation scores are available for the UI (see handle_question).
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
def search_documents(query: str) -> str:
    """Search the user's own uploaded/synced documents for information
    relevant to the query. Use this whenever the question could be about the
    user's own documents rather than general knowledge, math, or the time."""
    results = vector_store.similarity_search(query, k=2)
    if not results:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in results)

agent_tools = [calculate, get_current_time, search_documents]
agent_tools_by_name = {t.name: t for t in agent_tools}
agent_llm = agent_llm_base.bind_tools(agent_tools)

AGENT_SYSTEM_PROMPT = (
    "You have access to three tools: 'calculate' (arithmetic on numbers "
    "only), 'get_current_time' (current date/time only), and "
    "'search_documents' (search the user's own documents). "
    "For general knowledge not covered by these - capitals, common facts, "
    "definitions - call NO tool and answer directly. "
    "Calling a tool for a question it wasn't built for is wrong even if you "
    "end up guessing the right answer anyway. "
    "If a tool result starts with 'TOOL ERROR', do not guess or make up an "
    "answer - tell the user the tool failed and why."
)

# few-shot examples teach the routing decision far more reliably than the
# instruction text alone, especially for a small model like llama3.2:3b
AGENT_FEW_SHOT = [
    {"role": "user", "content": "What is 12 times 8?"},
    AIMessage(content="", tool_calls=[
        {"name": "calculate", "args": {"expression": "12*8"}, "id": "ex1"}
    ]),
    ToolMessage(content="96", tool_call_id="ex1"),
    AIMessage(content="12 times 8 is 96."),

    {"role": "user", "content": "What is the capital of France?"},
    AIMessage(content="The capital of France is Paris."),

    {"role": "user", "content": "What's today's date?"},
    AIMessage(content="", tool_calls=[
        {"name": "get_current_time", "args": {}, "id": "ex2"}
    ]),
    ToolMessage(content="2026-01-15 10:00:00", tool_call_id="ex2"),
    AIMessage(content="Today's date is 2026-01-15."),

    {"role": "user", "content": "What is the capital of Italy?"},
    AIMessage(content="The capital of Italy is Rome."),

    {"role": "user", "content": "What CPU does my gaming PC build use?"},
    AIMessage(content="", tool_calls=[
        {"name": "search_documents", "args": {"query": "CPU gaming PC build"}, "id": "ex3"}
    ]),
    ToolMessage(content="CPU: AMD Ryzen 7 7800X3D", tool_call_id="ex3"),
    AIMessage(content="Your gaming PC build uses an AMD Ryzen 7 7800X3D."),

    {"role": "user", "content": "What does my resume say about my skills?"},
    AIMessage(content="", tool_calls=[
        {"name": "search_documents", "args": {"query": "skills"}, "id": "ex4"}
    ]),
    ToolMessage(content="Skills: Python, SQL, Excel", tool_call_id="ex4"),
    AIMessage(content="Your resume lists Python, SQL, and Excel as skills."),
]

prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# NEW: total size in MB of everything inside a folder (walks subfolders too)
def get_folder_size_mb(path):
    if not os.path.isdir(path):
        return 0.0
    total_bytes = sum(
        os.path.getsize(os.path.join(root, f))
        for root, dirs, files in os.walk(path)
        for f in files
    )
    return total_bytes / (1024 * 1024)

# UPDATED: now takes an optional progress_callback(index, total, filename) so the UI can
# show a live progress bar instead of a plain spinner. Supports .txt, .md, .pdf, .docx and
# reports any file that couldn't be loaded, instead of it silently vanishing.
def sync_store(folder, progress_callback=None):
    filepaths = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            filepaths.append(os.path.join(root, filename))

    all_docs = []
    skipped = []
    for i, filepath in enumerate(filepaths):
        if progress_callback:
            progress_callback(i, len(filepaths), os.path.basename(filepath))
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

    result = index(chunks, record_manager, vector_store, cleanup="full", source_id_key="source", key_encoder="blake2b")
    return result, skipped

generation_chain = prompt | llm | StrOutputParser()

import re
import random

def _clean_suggestion_line(line):
    line = line.strip()
    # strip common list prefixes: "1. ", "1) ", "- ", "• "
    return re.sub(r"^(\d+[\.\)]\s*|[-•]\s*)", "", line).strip()

# UPDATED: renders the final suggestions as a plain static list (same look as while
# streaming), not clickable buttons - no more box-styled chips.
def render_suggestion_buttons(container, questions):
    if not questions:
        return
    with container:
        st.caption("Try asking:")
        st.markdown("\n\n".join(f"• {q}" for q in questions))

# UPDATED: streams the LLM's response token-by-token instead of waiting for the full
# reply, and renders each completed question as its own chip immediately (as soon as a
# line break arrives) - so chips pop in one at a time, like ChatGPT typing out an answer,
# instead of all 4 appearing together only once the whole generation is finished.
def generate_suggested_questions(container):
    # fetch ids only first (cheap - no document text/embeddings), then randomly
    # pick 5 and fetch just those - gives random variety each run without
    # pulling the whole store's text into memory
    all_ids = vector_store.get(include=[])["ids"]
    if not all_ids:
        return []

    sampled_ids = random.sample(all_ids, min(5, len(all_ids)))
    stored = vector_store.get(ids=sampled_ids)
    sample_text = "\n\n".join(stored["documents"])[:2000]

    collected = []   # fully completed lines
    buffer = ""      # current in-progress line, growing word by word

    # UPDATED: builds ONE combined markdown string (completed lines + the growing partial
    # line) and writes it with a SINGLE st.markdown() call, instead of a caption plus a
    # separate st.write() per bullet. Rewriting many separate elements on every token
    # caused a visible flicker (the whole set gets torn down and rebuilt each time) -
    # one single block updating its text is much smoother, no flicker.
    def render_progress():
        lines_md = "\n\n".join(f"• {q}" for q in collected)
        partial = _clean_suggestion_line(buffer)
        if partial:
            lines_md = (lines_md + "\n\n" if lines_md else "") + f"• {partial}"
        with container:
            st.caption("Try asking:")
            st.markdown(lines_md)

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
        render_progress()  # called every chunk, so the active line grows word by word

    if len(collected) < 2:
        line = _clean_suggestion_line(buffer)
        if line:
            collected.append(line)

    render_suggestion_buttons(container, collected[:2])
    return collected[:2]

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, sources, scored, elapsed) tuples
# UPDATED: None means "not generated yet" (vs [] meaning "generated, nothing to suggest") -
# generation is deferred until after the rest of the page has already rendered, so the slow
# LLM call doesn't block the whole page from showing up.
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = None

# NEW: shared logic for the "search documents" path - used by both Auto mode (when the
# agent picks this tool) and the explicit "Document Search" mode (which skips the
# decision step entirely and always searches).
def run_document_search(question, query):
    # Chroma's default score is a DISTANCE, not a similarity - LOWER = more relevant.
    scored = vector_store.similarity_search_with_score(query, k=2)
    retrieved_docs = [doc for doc, score in scored]
    cite_sources = sorted(set(doc.metadata["source"] for doc in retrieved_docs))
    context = format_docs(retrieved_docs)
    answer = st.write_stream(generation_chain.stream({"context": context, "question": question}))
    if "don't know" in answer.lower():
        cite_sources = []
    else:
        for doc, score in scored:
            with st.container(border=True):
                st.caption(f"📄 {doc.metadata['source']}")
                st.markdown(f"> {doc.page_content[:250]}...")
    return answer, cite_sources, scored

# NEW: shared logic for running calculate/get_current_time and phrasing the raw result
# into a natural sentence via the main (bigger) LLM.
def run_tool(tool_name, args, question):
    tool_fn = agent_tools_by_name[tool_name]
    result = tool_fn.invoke(args)
    if result.startswith("TOOL ERROR"):
        # don't trust the LLM to notice/respect this itself - surface it directly
        st.error(f"Tool failed: {result}")
        return f"Tool failed: {result}"
    phrase_prompt = (
        f"Question: {question}\nTool result: {result}\n\n"
        f"Answer the question naturally using this result, in one short sentence."
    )
    return st.write_stream(llm.stream(phrase_prompt))

MODE_AUTO = "Auto"
MODE_DOCS = "Documents"
MODE_CALC = "Calculator"
MODE_TIME = "Time"
MODE_DIRECT = "Direct"

# UPDATED: the mode picker (below, near the chat input) lets you force a specific path
# instead of trusting the small agent_llm's routing decision - useful since it sometimes
# over-triggers a tool for general-knowledge questions (a known small-model limitation).
# "Auto" keeps the original agent-decides behavior.
def handle_question(question):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        start = time.time()
        cite_sources = []
        scored = []
        mode = st.session_state.get("mode", MODE_AUTO)

        if mode == MODE_DIRECT:
            answer = st.write_stream(llm.stream(question))

        elif mode == MODE_DOCS:
            answer, cite_sources, scored = run_document_search(question, question)

        elif mode == MODE_CALC:
            # only offer the ONE tool - removes the ambiguity that causes over-triggering,
            # since there's nothing else for the model to (wrongly) pick instead
            calc_only_llm = agent_llm_base.bind_tools([calculate])
            decision = calc_only_llm.invoke([{"role": "user", "content": question}])
            if decision.tool_calls:
                call = decision.tool_calls[0]
                answer = run_tool(call["name"], call["args"], question)
            else:
                st.warning("Couldn't find a math expression in that question.")
                answer = "Couldn't find a math expression in that question."

        elif mode == MODE_TIME:
            answer = run_tool("get_current_time", {}, question)

        else:  # MODE_AUTO
            agent_messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *AGENT_FEW_SHOT,
                {"role": "user", "content": question},
            ]
            decision = agent_llm.invoke(agent_messages)

            if not decision.tool_calls:
                # no tool needed - general knowledge question, answer directly
                answer = st.write_stream(llm.stream(question))
            else:
                call = decision.tool_calls[0]  # keep it simple: act on the first request only
                if call["name"] == "search_documents":
                    answer, cite_sources, scored = run_document_search(question, call["args"]["query"])
                else:
                    answer = run_tool(call["name"], call["args"], question)

        elapsed = time.time() - start
        st.caption(f"⏱️ Answered in {elapsed:.2f}s")

        if scored:
            with st.expander("Retrieval details (scores)"):
                for doc, score in scored:
                    st.write(f"`{score:.4f}` — {doc.metadata['source']}")

    st.session_state.history.append((question, answer, cite_sources, scored, elapsed))

# NEW: reserved here (before the sidebar) so its position in the main column is fixed early,
# even though it may not get filled in until later (either immediately below, if cached, or
# from inside the sidebar's sync buttons, or at the very end of the script on first load).
suggestions_placeholder = st.empty()

with st.sidebar:
    st.header("Documents")

    # NEW: upload files directly instead of only typing a local folder path
    uploaded_files = st.file_uploader(
        "Upload files", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Sync uploaded files"):
        os.makedirs("uploaded_docs", exist_ok=True)
        for uf in uploaded_files:
            with open(os.path.join("uploaded_docs", uf.name), "wb") as f:
                f.write(uf.getbuffer())

        progress_bar = st.progress(0, text="Starting...")
        def update_progress(i, total, filename):
            progress_bar.progress((i + 1) / total, text=f"Processing {filename} ({i + 1}/{total})")
        result, skipped_files = sync_store("uploaded_docs", progress_callback=update_progress)
        progress_bar.empty()

        st.success(f"Added: {result['num_added']}, Updated: {result['num_updated']}, "
                   f"Skipped (unchanged): {result['num_skipped']}, Deleted: {result['num_deleted']}")
        if skipped_files:
            with st.expander(f"⚠️ {len(skipped_files)} file(s) skipped"):
                for msg in skipped_files:
                    st.write(f"- {msg}")
        st.session_state.suggested_questions = generate_suggested_questions(suggestions_placeholder)
        st.rerun()

    st.divider()
    st.caption("Or sync from a local folder path")
    folder = st.text_input("Folder path", value="docs")
    if st.button("Sync folder"):
        progress_bar = st.progress(0, text="Starting...")
        def update_progress(i, total, filename):
            progress_bar.progress((i + 1) / total, text=f"Processing {filename} ({i + 1}/{total})")
        result, skipped_files = sync_store(folder, progress_callback=update_progress)
        progress_bar.empty()

        st.success(f"Added: {result['num_added']}, Updated: {result['num_updated']}, "
                   f"Skipped (unchanged): {result['num_skipped']}, Deleted: {result['num_deleted']}")
        if skipped_files:
            with st.expander(f"⚠️ {len(skipped_files)} file(s) skipped"):
                for msg in skipped_files:
                    st.write(f"- {msg}")
        st.session_state.suggested_questions = generate_suggested_questions(suggestions_placeholder)
        st.rerun()

    stored = vector_store.get()
    sources_in_store = sorted(set(m["source"] for m in stored["metadatas"])) if stored["metadatas"] else []
    if sources_in_store:
        st.caption(f"Indexed: {len(sources_in_store)} file(s)")
        with st.expander("Indexed files"):
            for f in sources_in_store:
                # Since this app runs locally, a click can open the file in its default app.
                if st.button(f, key=f"open_{f}"):
                    os.startfile(f)

    # NEW: export the conversation as a downloadable text file
    if st.session_state.history:
        st.divider()
        transcript = "\n\n".join(
            f"You: {q}\nAI: {a}" for q, a, *_ in st.session_state.history
        )
        st.download_button("Export conversation", transcript, file_name="conversation.txt")

    # NEW: show disk usage of the vector database and uploaded files, and a button to
    # wipe all locally stored data (two-step confirm, since this is destructive/irreversible).
    st.divider()
    st.subheader("Local storage")
    chroma_size = get_folder_size_mb("chroma_db")
    uploads_size = get_folder_size_mb("uploaded_docs")
    st.write(f"Vector database: {chroma_size:.2f} MB")
    st.write(f"Uploaded files: {uploads_size:.2f} MB")
    st.write(f"Total: {chroma_size + uploads_size:.2f} MB")

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

    if not st.session_state.confirm_clear:
        if st.button("🗑️ Clear all local data"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("This deletes the vector database, the record manager, and all uploaded files. This cannot be undone.")
        col1, col2 = st.columns(2)
        if col1.button("Yes, delete everything"):
            import shutil
            if os.path.isdir("chroma_db"):
                shutil.rmtree("chroma_db")
            if os.path.isdir("uploaded_docs"):
                shutil.rmtree("uploaded_docs")
            if os.path.isfile("record_manager.db"):
                os.remove("record_manager.db")
            # cached objects point at now-deleted files, so clear them and start fresh
            st.cache_resource.clear()
            st.session_state.history = []
            st.session_state.suggested_questions = []
            st.session_state.confirm_clear = False
            st.rerun()
        if col2.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()

# If suggestions are already known (cached from a previous run), render them into the
# placeholder instantly - no need to re-stream something we already have.
if st.session_state.suggested_questions:
    render_suggestion_buttons(suggestions_placeholder, st.session_state.suggested_questions)

for question, answer, cite_sources, scored, elapsed in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if cite_sources:
            for doc, score in scored:
                with st.container(border=True):
                    st.caption(f"📄 {doc.metadata['source']}")
                    st.markdown(f"> {doc.page_content[:250]}...")
        st.caption(f"⏱️ Answered in {elapsed:.2f}s")
        if scored:
            with st.expander("Retrieval details (scores)"):
                for doc, score in scored:
                    st.write(f"`{score:.4f}` — {doc.metadata['source']}")

# NEW: st.bottom pins its contents to the bottom of the page, same as chat_input - and a
# selectbox (instead of radio buttons) renders as a compact dropdown, like ChatGPT's model
# picker, sitting right next to the input instead of taking up a full row above it.
with st.bottom:
    mode_col, input_col = st.columns([1, 4], vertical_alignment="bottom")
    with mode_col:
        st.selectbox(
            "Mode",
            [MODE_AUTO, MODE_DOCS, MODE_CALC, MODE_TIME, MODE_DIRECT],
            key="mode",
            label_visibility="collapsed",
        )
    with input_col:
        question = st.chat_input("Ask about your documents, do math, or ask the time...")
if question:
    handle_question(question)

# Only now, after everything above has already rendered and is visible to the user, do we
# generate the suggested questions (if we haven't yet) - streaming each one into the
# placeholder as it completes, instead of blocking the whole page until all 4 are ready.
if st.session_state.suggested_questions is None:
    st.session_state.suggested_questions = generate_suggested_questions(suggestions_placeholder)
