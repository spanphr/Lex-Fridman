import hashlib


def get_chunk_id(doc):
    """
    Generate a deterministic ID for an existing FAISS document.

    We use metadata + page content so we can uniquely identify
    chunks without rebuilding the current baseline index.
    """

    guest = doc.metadata.get("guest", "")
    title = doc.metadata.get("title", "")
    text = doc.page_content

    raw = f"{guest}|{title}|{text}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]
