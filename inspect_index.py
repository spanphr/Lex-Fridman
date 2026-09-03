from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from evals.utils import get_chunk_id


load_dotenv()

embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

docs = list(vector_store.docstore._dict.values())

print(f"Total chunks: {len(docs)}")

print("\n--- SAMPLE CHUNKS ---")

for i, doc in enumerate(docs[:5]):
    print(f"\nChunk {i + 1}")
    print(f"Chunk ID: {get_chunk_id(doc)}")
    print(f"Guest: {doc.metadata.get('guest')}")
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Episode ID: {doc.metadata.get('id')}")

    print("\nText:")
    print(doc.page_content[:500])

    print("-" * 80)