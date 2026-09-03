import re
import difflib
import urllib.parse

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama

from config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    OLLAMA_MODEL,
    FETCH_K,
    N_EPISODES,
    CHUNKS_PER_EPISODE,
)

# ============================================================
# Setup
# ============================================================

load_dotenv()

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# Build the set of valid guest names ONCE from the loaded FAISS docstore
# (in-memory; no CSV read per query).
VALID_GUESTS = sorted({
    (doc.metadata or {}).get("guest")
    for doc in vector_store.docstore._dict.values()
    if (doc.metadata or {}).get("guest")
})

# Common question / filler words excluded when matching name tokens and when
# collecting capitalized name candidates.
_STOPWORDS = {
    "what", "who", "how", "why", "when", "where", "does", "do", "did", "is",
    "are", "was", "were", "the", "a", "an", "in", "on", "about", "and", "of",
    "to", "think", "thinks", "thought", "say", "says", "said", "believe",
    "believes", "feel", "feels", "his", "her", "their", "lex", "fridman",
    "god", "ai",
}


# ============================================================
# Guest resolution
# ============================================================

def _capitalized_candidates(question):
    """Collect consecutive Title-Case tokens as candidate guest names.

    Used only to detect guest-scoped intent (and for the difflib typo
    fallback); it does NOT parse arbitrary grammar.
    """
    tokens = question.split()
    candidates = []
    current = []
    for i, tok in enumerate(tokens):
        clean = re.sub(r"[^A-Za-z]", "", tok)
        is_titlecase = (
            len(clean) >= 2
            and clean[:1].isupper()
            and clean[1:].islower()
        )
        if is_titlecase and i != 0 and clean.lower() not in _STOPWORDS:
            current.append(clean)
        else:
            if current:
                candidates.append(" ".join(current))
                current = []
    if current:
        candidates.append(" ".join(current))
    return candidates


def _resolve_guest(question):
    """Resolve a guest from the full question against VALID_GUESTS.

    Returns one of:
      ("resolved",  <guest name>)
      ("ambiguous", [<candidate guests>])
      ("not_found", <requested name>)   # guest-scoped but not in dataset
      ("none",      None)               # no guest detected -> global search
    """
    q_lower = question.lower()

    # Step 1: exact / full guest-name occurrence in the question.
    exact = [g for g in VALID_GUESTS if g.lower() in q_lower]
    if len(exact) == 1:
        return ("resolved", exact[0])
    if len(exact) > 1:
        return ("ambiguous", exact)

    # Step 2: surname / token match against VALID_GUESTS.
    q_tokens = {t for t in re.findall(r"[a-z]+", q_lower)
                if t not in _STOPWORDS}
    token_matches = []
    for g in VALID_GUESTS:
        g_tokens = [t for t in re.findall(r"[a-z]+", g.lower()) if len(t) >= 3]
        if any(t in q_tokens for t in g_tokens):
            token_matches.append(g)
    if len(token_matches) == 1:
        return ("resolved", token_matches[0])
    if len(token_matches) > 1:
        return ("ambiguous", token_matches)

    # Step 3: difflib typo fallback on capitalized name candidates.
    candidates = _capitalized_candidates(question)
    fuzzy = []
    for cand in candidates:
        match = difflib.get_close_matches(cand, VALID_GUESTS, n=1, cutoff=0.8)
        if match:
            fuzzy.append(match[0])
    fuzzy = list(dict.fromkeys(fuzzy))
    if len(fuzzy) == 1:
        return ("resolved", fuzzy[0])
    if len(fuzzy) > 1:
        return ("ambiguous", fuzzy)

    # No resolution: guest-scoped (a name was present) vs. no guest at all.
    if candidates:
        return ("not_found", ", ".join(candidates))
    return ("none", None)


# ============================================================
# Retrieval
# ============================================================

def retrieve_documents(query):
    """Return the TOP_K most similar transcript chunks for a query."""
    return vector_store.similarity_search(query, k=TOP_K)


# ============================================================
# Context building
# ============================================================

def build_context(documents):
    """Combine retrieved chunks into a single context string.

    Prepends useful metadata (guest, title, episode/id) before each
    chunk when available.
    """
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        guest = meta.get("guest")
        title = meta.get("title")
        episode = meta.get("episode") or meta.get("id")

        header_lines = [f"[Source {i}]"]
        if guest:
            header_lines.append(f"Guest: {guest}")
        if title:
            header_lines.append(f"Title: {title}")
        if episode:
            header_lines.append(f"Episode/ID: {episode}")

        header = "\n".join(header_lines)
        blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(blocks)


# ============================================================
# Answer generation
# ============================================================

def _debug_print_retrieval(question, guest=None):
    """Print retrieved chunks (with scores when available) for debugging.

    When `guest` is provided, retrieval is filtered to that guest's chunks.
    """
    scope = f"guest={guest!r}" if guest else "GLOBAL"
    print("=" * 60)
    print(f"RETRIEVAL DEBUG for: {question!r}  [{scope}]")
    print("=" * 60)

    # FAISS returns (Document, distance) pairs; lower distance = more similar.
    if guest is not None:
        scored = vector_store.similarity_search_with_score(
            question, k=TOP_K, filter={"guest": guest}, fetch_k=FETCH_K
        )
    else:
        scored = vector_store.similarity_search_with_score(question, k=TOP_K)

    if not scored:
        print("\n(no chunks retrieved)")
    for rank, (doc, score) in enumerate(scored, start=1):
        meta = doc.metadata or {}
        episode = meta.get("episode") or meta.get("id")
        print(f"\n----- Rank {rank} (distance={score:.4f}, lower is closer) -----")
        print(f"Guest      : {meta.get('guest')}")
        print(f"Title      : {meta.get('title')}")
        print(f"Episode/ID : {episode}")
        print(f"Chunk[:500]:\n{doc.page_content[:500]}")
    print("\n" + "=" * 60 + "\n")


def generate_answer(question):
    """Retrieve context and answer the question strictly from it.

    Guest-scoped questions ("What does <guest> say about <topic>?") are
    resolved to a known guest and retrieval is filtered to that guest.
    """
    status, value = _resolve_guest(question)

    if status == "ambiguous":
        return (
            "Your question matches multiple guests: "
            + ", ".join(value)
            + ". Please specify which one you mean."
        )

    if status == "not_found":
        return (
            f"The dataset has no episode with guest '{value}'. "
            "No global search was performed."
        )

    if status == "resolved":
        print(f"[guest-aware retrieval] resolved guest: {value!r}")
        _debug_print_retrieval(question, guest=value)
        documents = vector_store.similarity_search(
            question, k=TOP_K, filter={"guest": value}, fetch_k=FETCH_K
        )
    else:  # "none" -> preserve existing global behavior
        print("[guest-aware retrieval] no guest detected; using global search")
        _debug_print_retrieval(question)
        documents = retrieve_documents(question)

    context = build_context(documents)

    prompt = (
        "You are a helpful assistant answering questions about the "
        "Lex Fridman podcast using only the transcript excerpts provided "
        "below.\n\n"
        "Rules:\n"
        "- Answer ONLY using the information in the context.\n"
        "- If the answer is not contained in the context, say you do not "
        "know.\n"
        "- Do not use outside knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)
    return response.content


# ============================================================
# Explore Episodes mode
# ============================================================

def _youtube_search_url(guest, title):
    """Build a YouTube *search* URL (not a direct video link) for an episode."""
    query = f"Lex Fridman {guest} {title}".strip()
    return (
        "https://www.youtube.com/results?"
        + urllib.parse.urlencode({"search_query": query})
    )


def retrieve_grouped(query, guest=None):
    """Retrieve a wide pool and group it into distinct episodes.

    Returns a list of episode groups (best-first), each shaped like:

        {
            "row": <unique episode key from metadata["row"]>,
            "episode_id": <metadata["id"], not unique>,
            "guest": ...,
            "title": ...,
            "relevance": <best/lowest FAISS distance in the episode>,
            "chunks": [<Document>, ...],   # up to CHUNKS_PER_EPISODE, best-first
        }

    Grouping key is metadata["row"] (unique), NOT id (id 14 is duplicated).
    """
    search_kwargs = {"k": FETCH_K, "fetch_k": FETCH_K}
    if guest is not None:
        search_kwargs["filter"] = {"guest": guest}

    # (Document, distance) pairs; lower distance = more similar.
    scored = vector_store.similarity_search_with_score(query, **search_kwargs)

    groups = {}
    for doc, score in scored:
        meta = doc.metadata or {}
        row = meta.get("row")
        if row is None:
            continue
        group = groups.get(row)
        if group is None:
            group = {
                "row": row,
                "episode_id": meta.get("id"),
                "guest": meta.get("guest"),
                "title": meta.get("title"),
                "relevance": score,
                "chunks": [],
            }
            groups[row] = group
        # scored is already distance-ascending, so the first seen is the best.
        group["relevance"] = min(group["relevance"], score)
        if len(group["chunks"]) < CHUNKS_PER_EPISODE:
            group["chunks"].append(doc)

    ranked = sorted(groups.values(), key=lambda g: g["relevance"])
    selected = ranked[:N_EPISODES]

    # Top-up: enrich each SELECTED episode with an episode-scoped retrieval so
    # the card has richer context. This does NOT change which episodes were
    # selected, nor their ranking relevance (that stays the wide-pool best).
    # FAISS metadata filtering is post-retrieval, so fetch_k must exceed
    # CHUNKS_PER_EPISODE; a bounded value (not the whole index) keeps this
    # cheap and scalable while still surfacing a selected episode's top chunks.
    topup_fetch_k = 200
    for group in selected:
        group["chunks_before_topup"] = len(group["chunks"])
        scoped = vector_store.similarity_search(
            query,
            k=CHUNKS_PER_EPISODE,
            filter={"row": group["row"]},
            fetch_k=topup_fetch_k,
        )
        if scoped:
            group["chunks"] = scoped  # relevance score is intentionally kept

    return selected


def answer_episode(question, episode_group):
    """Answer the question grounded ONLY in a single episode's chunks."""
    context = build_context(episode_group["chunks"])

    prompt = (
        "You are answering a question using ONLY the transcript excerpts "
        "from a single Lex Fridman podcast episode shown below.\n\n"
        f"Episode guest: {episode_group.get('guest')}\n"
        f"Episode title: {episode_group.get('title')}\n\n"
        "Rules:\n"
        "- Answer ONLY using the excerpts from THIS episode.\n"
        "- Be concise: 2-4 sentences suitable for a website card.\n"
        "- If these excerpts do not clearly address the question, say that "
        "this episode does not clearly address the question.\n"
        "- Do not use outside knowledge.\n\n"
        f"Excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)
    return response.content


def explore(question):
    """Explore Episodes backend: one grounded answer per distinct episode.

    Returns a structured dict (never depends on printed output):

        {"status": "ok", "query", "resolved_guest", "episodes": [...]}
        {"status": "guest_not_found", "query", "requested_guest"}
        {"status": "ambiguous_guest", "query", "candidates": [...]}
    """
    status, value = _resolve_guest(question)

    if status == "ambiguous":
        return {
            "status": "ambiguous_guest",
            "query": question,
            "candidates": value,
        }

    if status == "not_found":
        return {
            "status": "guest_not_found",
            "query": question,
            "requested_guest": value,
        }

    resolved_guest = value if status == "resolved" else None
    groups = retrieve_grouped(question, guest=resolved_guest)

    episodes = []
    for group in groups:
        answer = answer_episode(question, group)
        episodes.append({
            "row": group["row"],
            "episode_id": group["episode_id"],
            "guest": group["guest"],
            "title": group["title"],
            "answer": answer,
            "video_url": _youtube_search_url(group["guest"], group["title"]),
            "relevance": group["relevance"],
            "chunks_used": [
                {
                    "row": (c.metadata or {}).get("row"),
                    "id": (c.metadata or {}).get("id"),
                    "snippet": c.page_content[:200],
                }
                for c in group["chunks"]
            ],
        })

    return {
        "status": "ok",
        "query": question,
        "resolved_guest": resolved_guest,
        "episodes": episodes,
    }


# ============================================================
# Conversation mode
# ============================================================

# How many recent history messages to include in the prompt (short-term only).
_HISTORY_TURNS = 4


# How many recent USER messages from history feed the retrieval query.
_RETRIEVAL_USER_TURNS = 2


def build_conversation_retrieval_query(question, history=None):
    """Build the retrieval query from recent USER turns + current question.

    Deterministic (no LLM). Includes at most the last two USER messages from
    history (assistant messages are ignored), in chronological order, followed
    by the current question. This keeps follow-ups like "how do their views
    differ?" anchored to the original topic.
    """
    user_msgs = [
        (msg.get("content") or "").strip()
        for msg in (history or [])
        if (msg.get("role") == "user") and (msg.get("content") or "").strip()
    ]
    recent_users = user_msgs[-_RETRIEVAL_USER_TURNS:]
    parts = recent_users + [question.strip()]
    return " ".join(p for p in parts if p)


def _format_history(history):
    """Render the last few history messages as plain text for the prompt."""
    if not history:
        return ""
    recent = history[-_HISTORY_TURNS:]
    lines = []
    for msg in recent:
        role = (msg.get("role") or "user").capitalize()
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def conversation_answer(question, history=None):
    """Grounded multi-episode conversation turn.

    Reuses guest resolution + retrieve_grouped() as the retrieval backend.
    Returns:
        {"status": "ok", "answer", "resolved_guest", "sources": [...]}
        {"status": "guest_not_found", "requested_guest", ...}
        {"status": "ambiguous_guest", "candidates", ...}
    """
    status, value = _resolve_guest(question)

    if status == "ambiguous":
        return {
            "status": "ambiguous_guest",
            "query": question,
            "candidates": value,
        }

    if status == "not_found":
        return {
            "status": "guest_not_found",
            "query": question,
            "requested_guest": value,
        }

    resolved_guest = value if status == "resolved" else None

    # Retrieval query folds in recent user turns so follow-ups stay on-topic;
    # guest resolution above still uses only the current question.
    retrieval_query = build_conversation_retrieval_query(question, history)
    groups = retrieve_grouped(retrieval_query, guest=resolved_guest)

    # Grounding context: all selected episodes' chunks, labelled by episode.
    context_docs = []
    for group in groups:
        context_docs.extend(group["chunks"])
    context = build_context(context_docs)

    # Deduplicate sources by episode row (retrieve_grouped already yields
    # distinct rows, but stay defensive).
    sources = []
    seen_rows = set()
    for group in groups:
        row = group["row"]
        if row in seen_rows:
            continue
        seen_rows.add(row)
        sources.append({
            "row": row,
            "episode_id": group["episode_id"],
            "guest": group["guest"],
            "title": group["title"],
        })

    history_text = _format_history(history)
    history_block = (
        f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    )

    prompt = (
        "You are having a grounded conversation about the Lex Fridman podcast. "
        "Use ONLY the transcript excerpts below, which come from several "
        "different episodes and guests.\n\n"
        "Rules:\n"
        "- Base every claim on the excerpts; never state anything not "
        "supported by them.\n"
        "- When the excerpts cover multiple guests, synthesize across them, "
        "and compare or contrast their viewpoints when the question asks for "
        "it.\n"
        "- Attribute views to the specific guest when possible.\n"
        "- If the excerpts do not support an answer, say so.\n"
        "- Keep the tone conversational.\n\n"
        f"{history_block}"
        f"Episode excerpts:\n{context}\n\n"
        f"Current question: {question}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)

    return {
        "status": "ok",
        "answer": response.content,
        "resolved_guest": resolved_guest,
        "sources": sources,
    }


# ============================================================
# Terminal test
# ============================================================

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer = generate_answer(question)
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
