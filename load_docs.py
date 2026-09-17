from langchain_unstructured import UnstructuredLoader
import glob

file_paths = glob.glob("docs/**/*.txt", recursive=True) + glob.glob("docs/**/*.pdf", recursive=True)

loader = UnstructuredLoader(file_path=file_paths, strategy="fast", chunking_strategy="by_title")
chunks = loader.load()   # these are already properly-sized chunks, ready for embedding

print(f"Loaded {len(chunks)} chunks total")
for c in chunks:
    print(f"\nSource: {c.metadata['source']}")
    print(f"Content preview: {c.page_content[:150]}...")