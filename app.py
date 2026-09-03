import streamlit as st

from rag import conversation_answer


st.set_page_config(page_title="Lex Fridman RAG", page_icon="🎙️")

st.title("Lex Fridman Podcast RAG")
st.caption("Ask a question and get an answer grounded in podcast transcripts.")

question = st.text_input(
    "Your question",
    placeholder="What do different guests say about artificial intelligence?",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching episodes and generating an answer..."):
        result = conversation_answer(question.strip())

    if result["status"] == "guest_not_found":
        st.warning(
            f"The dataset has no episode with guest "
            f"'{result['requested_guest']}'."
        )
    elif result["status"] == "ambiguous_guest":
        candidates = ", ".join(result["candidates"])
        st.warning(f"Guest name is ambiguous. Did you mean: {candidates}?")
    else:
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for source in result["sources"]:
            st.markdown(
                f"**Episode #{source['episode_id']}: {source['title']}**  \n"
                f"Guest: {source['guest']}"
            )
