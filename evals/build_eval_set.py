import random

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from evals.utils import get_chunk_id


load_dotenv()

random.seed(42)

vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

docs = list(vector_store.docstore._dict.values())

print(f"Total chunks: {len(docs)}")


# --------------------------------------------------
# Basic corpus statistics
# --------------------------------------------------

guests = {}

for doc in docs:
    guest = doc.metadata.get("guest", "Unknown")

    if guest not in guests:
        guests[guest] = []

    guests[guest].append(doc)


print(f"Total guests: {len(guests)}")

print("\nTop guests by chunks:")

for guest, guest_docs in sorted(
    guests.items(),
    key=lambda x: len(x[1]),
    reverse=True
)[:20]:
    print(f"{guest:35} {len(guest_docs)}")


# --------------------------------------------------
# Sample candidate chunks
# --------------------------------------------------

print("\n\n--- CANDIDATE EVALUATION CHUNKS ---")

eligible_guests = [
    guest
    for guest, guest_docs in guests.items()
    if len(guest_docs) >= 5
]

sampled_guests = random.sample(
    eligible_guests,
    min(10, len(eligible_guests))
)

for guest in sampled_guests:

    guest_docs = guests[guest]

    doc = random.choice(guest_docs)

    print("\n" + "=" * 80)
    print(f"Guest: {guest}")
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Chunk ID: {get_chunk_id(doc)}")

    print("\nTEXT:")
    print(doc.page_content[:1200])
