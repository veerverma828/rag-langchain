#for llm
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#Main backend
from fastapi import FastAPI
from pydantic import BaseModel
#for serving files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


#main llm setup
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="qwen2.5-coder:7b")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:""")

generation_chain = prompt | llm | StrOutputParser()

#routings-
app = FastAPI()

@app.get("/")
def read_index():
    return FileResponse("static/index.html")
#for static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    scored = vector_store.similarity_search_with_score(q.question, k=2)
    docs = [doc for doc, score in scored]
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = generation_chain.invoke({"context": context, "question": q.question})

    sources = [
        {"source": doc.metadata["source"], "score": score, "preview": doc.page_content[:200]}
        for doc, score in scored
    ]
    return {"answer": answer, "sources": sources}