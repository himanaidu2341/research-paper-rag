import streamlit as st
import tempfile
from rag_pipeline import build_rag_system

st.set_page_config(page_title="Research Paper QA", page_icon="📄")
st.title("Research Paper Question Answering System")

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if "chain" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.chain = build_rag_system(tmp_path)
        st.success("Ready! Ask your questions.")

    query = st.text_input("Ask a question about the paper:")
    if query:
        response = st.session_state.chain.invoke({"input": query})
        st.subheader("Answer:")
        st.write(response["answer"])

        with st.expander("Sources"):
            for doc in response["context"]:
                st.markdown(f"**Page {doc.metadata.get('page', 'Unknown')}**")
                st.write(doc.page_content[:300] + "...")
