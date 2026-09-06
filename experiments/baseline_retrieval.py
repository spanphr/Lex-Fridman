import json
import time

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langsmith import traceable

from evals.utils import get_chunk_id


# ============================================================
# Setup
# ============================================================

load_dotenv()

KS = [1, 3, 5, 10]


# ============================================================
# Load benchmark
# ============================================================

with open("evals/questions.json", "r") as f:
    questions = json.load(f)


# ============================================================
# Load baseline vector store
# ============================================================

embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)


# ============================================================
# LangSmith-traced retrieval function
# ============================================================

@traceable(
    name="baseline_retrieval",
    run_type="retriever",
)
def retrieve_question(item):
    """
    Retrieve top-k chunks for one benchmark question.

    LangSmith will capture:
    - question
    - gold chunk IDs
    - retrieved chunks
    - ranks
    - latency
    """

    question = item["question"]
    gold_ids = set(item["gold_chunk_ids"])

    start = time.perf_counter()

    results = vector_store.similarity_search(
        question,
        k=max(KS),
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    retrieved = []

    for rank, doc in enumerate(results, start=1):
        chunk_id = get_chunk_id(doc)

        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "guest": doc.metadata.get("guest"),
                "title": doc.metadata.get("title"),
                "episode_id": doc.metadata.get("id"),
                "is_gold": chunk_id in gold_ids,
                "text": doc.page_content[:500],
            }
        )

    found_rank = None

    for result in retrieved:
        if result["is_gold"]:
            found_rank = result["rank"]
            break

    return {
        "question_id": item["id"],
        "question": question,
        "guest": item.get("guest"),
        "title": item.get("title"),
        "category": item.get("category"),
        "gold_chunk_ids": list(gold_ids),
        "found_rank": found_rank,
        "latency_ms": elapsed_ms,
        "retrieved": retrieved,
    }


# ============================================================
# Warm-up
# ============================================================

print("\nWarming up embedding + retrieval pipeline...")

_ = vector_store.similarity_search(
    "warm up query",
    k=1,
)

print("Warm-up complete.\n")


# ============================================================
# Experiment
# ============================================================

hits = {k: 0 for k in KS}
reciprocal_ranks = []
latencies_ms = []

print("=== BASELINE RETRIEVAL EXPERIMENT ===\n")


for item in questions:
    run = retrieve_question(item)

    found_rank = run["found_rank"]
    elapsed_ms = run["latency_ms"]

    gold_ids = set(run["gold_chunk_ids"])

    retrieved_ids = [
        result["chunk_id"]
        for result in run["retrieved"]
    ]

    # --------------------------------------------------------
    # Hit@K
    # --------------------------------------------------------

    for k in KS:
        top_k = retrieved_ids[:k]

        if any(
            chunk_id in gold_ids
            for chunk_id in top_k
        ):
            hits[k] += 1

    # --------------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------------

    rr = 0.0

    if found_rank is not None:
        rr = 1.0 / found_rank

    reciprocal_ranks.append(rr)
    latencies_ms.append(elapsed_ms)

    # --------------------------------------------------------
    # Per-question output
    # --------------------------------------------------------

    print(
        f"{item['id']} | "
        f"rank={found_rank} | "
        f"latency={elapsed_ms:.2f} ms | "
        f"{item['question']}"
    )

    # --------------------------------------------------------
    # Inspect failure cases
    # --------------------------------------------------------

    if item["id"] in {"ret_001", "ret_004"}:
        print("\nTop retrieved chunks:")

        for result in run["retrieved"][:5]:
            print(
                f"{result['rank']}. "
                f"{result['guest']} | "
                f"{result['title']} | "
                f"{result['chunk_id']} | "
                f"gold={result['is_gold']}"
            )

            print(
                result["text"]
                .replace("\n", " ")
                [:300]
            )

            print()

        print("-" * 80)


# ============================================================
# Aggregate metrics
# ============================================================

n = len(questions)

print("\n=== RESULTS ===\n")

for k in KS:
    hit_rate = hits[k] / n

    print(
        f"Hit@{k:<2} "
        f"{hits[k]}/{n} "
        f"({hit_rate:.1%})"
    )


# ============================================================
# MRR
# ============================================================

mrr = sum(reciprocal_ranks) / n


# ============================================================
# Latency metrics
# ============================================================

avg_latency = sum(latencies_ms) / n

sorted_latencies = sorted(latencies_ms)

middle = len(sorted_latencies) // 2

if len(sorted_latencies) % 2 == 0:
    p50_latency = (
        sorted_latencies[middle - 1]
        + sorted_latencies[middle]
    ) / 2
else:
    p50_latency = sorted_latencies[middle]


print(f"\nMRR: {mrr:.3f}")

print(
    f"Average retrieval latency: "
    f"{avg_latency:.2f} ms"
)

print(
    f"P50 retrieval latency: "
    f"{p50_latency:.2f} ms"
)


# ============================================================
# Category breakdown
# ============================================================

print("\n=== CATEGORY BREAKDOWN ===\n")

categories = {}

for item in questions:
    category = item.get(
        "category",
        "unknown",
    )

    if category not in categories:
        categories[category] = 0

    categories[category] += 1

for category, count in categories.items():
    print(
        f"{category:<20} "
        f"{count} question(s)"
    )


print("\nExperiment complete.")