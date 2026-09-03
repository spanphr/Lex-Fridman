from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, TOP_K

load_dotenv()

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

query = "What is artificial intelligence?"

results = vector_store.similarity_search(query, k=TOP_K)

print(f"Retrieved {len(results)} results\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}")
    print(f"Guest: {doc.metadata.get('guest')}")
    print(f"Title: {doc.metadata.get('title')}")
    print(doc.page_content[:300])
    print("-" * 80)