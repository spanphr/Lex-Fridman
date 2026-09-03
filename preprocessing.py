import csv
from dotenv import load_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# Configuration
# ============================================================

csv.field_size_limit(131072 * 10)
load_dotenv()

# ============================================================
# (1) DOCUMENT LOADING
# ============================================================

loader = CSVLoader(
    file_path="data/podcastdata_dataset.csv",
    source_column="text",
    metadata_columns=["id", "guest", "title"],
)

docs = loader.load()

print("=" * 60)
print(f"Loaded {len(docs)} documents")
print("=" * 60)

print("\nFirst document preview:\n")
print(str(docs[0])[:300] + "...")

# ============================================================
# (2) DOCUMENT SPLITTING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

all_splits = text_splitter.split_documents(docs)

print("\n" + "=" * 60)
print(f"Created {len(all_splits)} chunks")
print("=" * 60)

print("\nEnd of first chunk:\n")
print(all_splits[0].page_content[-250:])

print("\nStart of second chunk:\n")
print(all_splits[1].page_content[:250])

# ============================================================
# (3) EMBEDDING & VECTOR STORE
# ============================================================

print("\n" + "=" * 60)
print("Creating embeddings...")
print("=" * 60)

embeddings = OpenAIEmbeddings()

vector_store = FAISS.from_documents(
    documents=all_splits,
    embedding=embeddings,
)

# Save the FAISS index
vector_store.save_local("faiss_index")

print("\n✅ FAISS index saved successfully!")

# ============================================================
# (4) TEST RETRIEVAL
# ============================================================

query = "What is artificial intelligence?"

results = vector_store.similarity_search(
    query=query,
    k=5,
)

print("\n" + "=" * 60)
print(f"Query: {query}")
print(f"Retrieved {len(results)} chunks")
print("=" * 60)

for i, doc in enumerate(results, start=1):
    print(f"\n----- Result {i} -----")
    print(f"Guest : {doc.metadata.get('guest')}")
    print(f"Title : {doc.metadata.get('title')}")
    print(f"Text  :\n{doc.page_content[:300]}...")