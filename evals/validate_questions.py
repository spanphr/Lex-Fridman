import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from evals.utils import get_chunk_id


load_dotenv()

vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,
)

docs = list(vector_store.docstore._dict.values())

chunk_lookup = {
    get_chunk_id(doc): doc
    for doc in docs
}

with open("evals/questions.json", "r") as f:
    questions = json.load(f)

print(f"Questions: {len(questions)}")
print(f"Indexed chunks: {len(chunk_lookup)}")

errors = 0

for item in questions:
    for chunk_id in item["gold_chunk_ids"]:

        if chunk_id not in chunk_lookup:
            print(
                f"❌ {item['id']} references missing chunk "
                f"{chunk_id}"
            )
            errors += 1
        else:
            doc = chunk_lookup[chunk_id]

            print(
                f"✅ {item['id']} | "
                f"{doc.metadata.get('guest')} | "
                f"{chunk_id}"
            )

if errors:
    print(f"\nValidation failed with {errors} error(s).")
else:
    print("\nAll gold chunk IDs are valid.")
