from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5-coder:7b")
response = llm.invoke("Say hello in one sentence.")
print(response.content)