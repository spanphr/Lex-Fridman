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
# Load vector store
# ============================================================

embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)


# ============================================================
# Retrieval functions
# ============================================================

@traceable(
    name="global_retrieval",
    run_type="retriever",
    tags=["experiment-1", "global", "faiss"],
)
def global_retrieve(item):
    """
    Current baseline:
    search across the entire corpus.
    """

    start = time.perf_counter()

    docs = vector_store.similarity_search(
        item["question"],
        k=max(KS),
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "question_id": item["id"],
        "guest": item["guest"],
        "latency_ms": latency_ms,
        "results": [
            {
                "rank": rank,
                "chunk_id": get_chunk_id(doc),
                "guest": doc.metadata.get("guest"),
                "title": doc.metadata.get("title"),
                "text": doc.page_content[:300],
            }
            for rank, doc in enumerate(docs, start=1)
        ],
    }


@traceable(
    name="guest_aware_retrieval",
    run_type="retriever",
    tags=["experiment-1", "guest-aware", "faiss"],
)
def guest_retrieve(item):
    """
    Experimental variant:
    use guest metadata to restrict candidate chunks.

    fetch_k is deliberately larger than k because LangChain's
    FAISS metadata filtering may filter candidates after the
    vector search.
    """

    start = time.perf_counter()

    docs = vector_store.similarity_search(
        item["question"],
        k=max(KS),
        filter={
            "guest": item["guest"],
        },
        fetch_k=1000,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "question_id": item["id"],
        "guest": item["guest"],
        "latency_ms": latency_ms,
        "results": [
            {
                "rank": rank,
                "chunk_id": get_chunk_id(doc),
                "guest": doc.metadata.get("guest"),
                "title": doc.metadata.get("title"),
                "text": doc.page_content[:300],
            }
            for rank, doc in enumerate(docs, start=1)
        ],
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_run(run, item):
    gold_ids = set(item["gold_chunk_ids"])

    retrieved_ids = [
        result["chunk_id"]
        for result in run["results"]
    ]

    hit_at_k = {}

    for k in KS:
        hit_at_k[k] = any(
            chunk_id in gold_ids
            for chunk_id in retrieved_ids[:k]
        )

    found_rank = None

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in gold_ids:
            found_rank = rank
            break

    reciprocal_rank = (
        1.0 / found_rank
        if found_rank is not None
        else 0.0
    )

    return {
        "found_rank": found_rank,
        "reciprocal_rank": reciprocal_rank,
        "hit_at_k": hit_at_k,
    }


def summarize(name, evaluations, latencies):
    n = len(evaluations)

    print(f"\n=== {name.upper()} ===\n")

    for k in KS:
        hits = sum(
            evaluation["hit_at_k"][k]
            for evaluation in evaluations
        )

        print(
            f"Hit@{k:<2} "
            f"{hits}/{n} "
            f"({hits / n:.1%})"
        )

    mrr = sum(
        evaluation["reciprocal_rank"]
        for evaluation in evaluations
    ) / n

    avg_latency = sum(latencies) / n

    sorted_latencies = sorted(latencies)

    middle = len(sorted_latencies) // 2

    if len(sorted_latencies) % 2 == 0:
        p50 = (
            sorted_latencies[middle - 1]
            + sorted_latencies[middle]
        ) / 2
    else:
        p50 = sorted_latencies[middle]

    print(f"\nMRR: {mrr:.3f}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P50 latency: {p50:.2f} ms")


# ============================================================
# Warm-up
# ============================================================

print("\nWarming up...")

_ = vector_store.similarity_search(
    "warm up query",
    k=1,
)

print("Warm-up complete.")


# ============================================================
# Run experiment
# ============================================================

global_evaluations = []
guest_evaluations = []

global_latencies = []
guest_latencies = []


print("\n=== EXPERIMENT 1: GLOBAL VS GUEST-AWARE ===\n")


for item in questions:

    global_run = global_retrieve(item)
    guest_run = guest_retrieve(item)

    global_eval = evaluate_run(
        global_run,
        item,
    )

    guest_eval = evaluate_run(
        guest_run,
        item,
    )

    global_evaluations.append(global_eval)
    guest_evaluations.append(guest_eval)

    global_latencies.append(
        global_run["latency_ms"]
    )

    guest_latencies.append(
        guest_run["latency_ms"]
    )

    print(
        f"{item['id']} | "
        f"guest={item['guest']} | "
        f"global_rank={global_eval['found_rank']} | "
        f"guest_rank={guest_eval['found_rank']}"
    )

    # Show changed/failure cases
    if (
        global_eval["found_rank"]
        != guest_eval["found_rank"]
    ):
        print("  Guest-aware top 3:")

        for result in guest_run["results"][:3]:
            print(
                f"    {result['rank']}. "
                f"{result['guest']} | "
                f"{result['chunk_id']}"
            )

        print()


# ============================================================
# Final comparison
# ============================================================

summarize(
    "Global retrieval",
    global_evaluations,
    global_latencies,
)

summarize(
    "Guest-aware retrieval",
    guest_evaluations,
    guest_latencies,
)


print("\n=== DELTA ===\n")

global_mrr = sum(
    x["reciprocal_rank"]
    for x in global_evaluations
) / len(global_evaluations)

guest_mrr = sum(
    x["reciprocal_rank"]
    for x in guest_evaluations
) / len(guest_evaluations)

print(
    f"MRR change: "
    f"{global_mrr:.3f} -> "
    f"{guest_mrr:.3f} "
    f"({guest_mrr - global_mrr:+.3f})"
)

print("\nExperiment complete.")
