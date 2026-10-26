
import streamlit as st
import tempfile
import os

from pdf_reader import extract_text_from_pdf
from text_chunker import split_text_into_chunks
from semantic_search import create_embeddings, search_chunks
from answer_generator import generate_answer


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Smart PDF Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chunks" not in st.session_state:
    st.session_state["chunks"] = None

if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None

if "pdf_names" not in st.session_state:
    st.session_state["pdf_names"] = []


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📚 PDF Assistant")
    st.caption("Your personal AI study companion")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload your study PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files.",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF(s) selected")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

        if st.button(
            "⚙️ Process PDFs",
            use_container_width=True,
            type="primary",
        ):
            all_chunks = []
            processed_files = []
            failed_files = []
            temp_paths = []

            try:
                with st.spinner("Processing your PDFs..."):
                    for uploaded_file in uploaded_files:
                        temp_path = None

                        try:
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                suffix=".pdf",
                            ) as temp_file:
                                temp_file.write(
                                    uploaded_file.getvalue()
                                )
                                temp_path = temp_file.name
                                temp_paths.append(temp_path)

                            text = extract_text_from_pdf(temp_path)

                            if not text or not text.strip():
                                failed_files.append(
                                    f"{uploaded_file.name}: "
                                    "No selectable text found"
                                )
                                continue

                            chunks = split_text_into_chunks(
                                text,
                                chunk_size=1000,
                                overlap=200,
                            )

                            if not chunks:
                                failed_files.append(
                                    f"{uploaded_file.name}: "
                                    "No text chunks created"
                                )
                                continue

                            # Keep the source filename with every chunk
                            for chunk in chunks:
                                all_chunks.append({
                                    "text": chunk,
                                    "source": uploaded_file.name,
                                })

                            processed_files.append(
                                uploaded_file.name
                            )

                        except Exception as error:
                            failed_files.append(
                                f"{uploaded_file.name}: {error}"
                            )

                        finally:
                            if (
                                temp_path
                                and os.path.exists(temp_path)
                            ):
                                os.remove(temp_path)
                                if temp_path in temp_paths:
                                    temp_paths.remove(temp_path)

                    if all_chunks:
                        embeddings = create_embeddings(all_chunks)

                        st.session_state["chunks"] = all_chunks
                        st.session_state["embeddings"] = embeddings
                        st.session_state["pdf_names"] = processed_files
                        st.session_state["messages"] = []

                        st.success(
                            f"Processed {len(processed_files)} PDF(s) "
                            f"into {len(all_chunks)} chunks!"
                        )

                        for error in failed_files:
                            st.warning(error)

                        st.rerun()

                    else:
                        st.error(
                            "No usable text was found in the "
                            "uploaded PDFs. Please try another file."
                        )

                        for error in failed_files:
                            st.warning(error)

            except Exception as error:
                st.error(f"Could not process PDFs: {error}")

            finally:
                for path in temp_paths:
                    if os.path.exists(path):
                        os.remove(path)

    st.divider()

    if st.session_state["chunks"] is not None:
        st.markdown("### 📄 Processed documents")

        for pdf_name in st.session_state["pdf_names"]:
            st.write(f"📘 {pdf_name}")

        st.caption(
            f"Total chunks: {len(st.session_state['chunks'])}"
        )

        if st.button(
            "🗑️ Clear chat",
            use_container_width=True,
        ):
            st.session_state["messages"] = []
            st.rerun()

    else:
        st.info("Upload and process your PDFs to get started.")

    st.divider()
    st.caption("Powered by semantic search and Gemini AI")


# ---------------- MAIN CHAT AREA ----------------

st.title("💬 Smart PDF Study Assistant")

st.write(
    "Ask questions across your uploaded PDFs "
    "and explore their content with AI-powered answers."
)

if st.session_state["chunks"] is None:
    st.markdown("### 👋 Welcome!")

    st.write(
        "Upload one or more PDFs from the sidebar, "
        "click **Process PDFs**, and ask your first question."
    )

    st.markdown("#### You can ask things like:")
    st.markdown(
        "- What is this document about?\n"
        "- Compare information across my PDFs.\n"
        "- List the important points.\n"
        "- Which document discusses a particular topic?"
    )

else:
    st.caption(
        f"📚 Studying {len(st.session_state['pdf_names'])} PDF(s)"
    )

    st.divider()

    # Display conversation history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander("📖 View relevant PDF passages"):
                    for index, source in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        st.markdown(
                            f"**Passage {index} — "
                            f"Source: {source.get('source', 'Unknown')} — "
                            f"Similarity: {source['score']:.3f}**"
                        )

                        st.write(source["text"])

                        if index < len(message["sources"]):
                            st.divider()

    # Chat input
    question = st.chat_input(
        "Ask a question about your PDFs..."
    )

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Searching your PDFs and generating an answer..."
                ):
                    results = search_chunks(
                        question,
                        st.session_state["chunks"],
                        st.session_state["embeddings"],
                        top_k=5,
                    )

                    # Map retrieved text back to its source PDF
                    source_lookup = {
                        chunk["text"]: chunk["source"]
                        for chunk in st.session_state["chunks"]
                    }

                    for result in results:
                        result["source"] = source_lookup.get(
                            result["text"],
                            "Unknown",
                        )

                    context = "\n\n".join(
                        f"Source PDF: {result['source']}\n"
                        f"{result['text']}"
                        for result in results
                    )

                    answer = generate_answer(question, context)

                st.markdown(answer)

                if results:
                    with st.expander(
                        "📖 View relevant PDF passages"
                    ):
                        for index, result in enumerate(
                            results,
                            start=1,
                        ):
                            st.markdown(
                                f"**Passage {index} — "
                                f"Source: {result['source']} — "
                                f"Similarity: {result['score']:.3f}**"
                            )

                            st.write(result["text"])

                            if index < len(results):
                                st.divider()

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": results,
                })

            except Exception as error:
                error_message = (
                    "Sorry, I couldn't generate an answer. "
                    f"Error: {error}"
                )

                st.error(error_message)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": error_message,
                })
