import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# ========== ENVIRONMENT SETUP ==========
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="📄 PDF Q&A Assistant", layout="wide")
st.title("📘 PDF Question-Answer App (LM Studio + LangChain)")
st.markdown("Ask questions from your uploaded PDF powered by **TinyLlama** in LM Studio!")

# ========== PDF UPLOAD ==========
uploaded_pdf = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    pdf_path = os.path.join("uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    # Step 1: Load and chunk
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    st.success(f"✅ Loaded and split into {len(chunks)} chunks")

    # Step 2: Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Step 3: Initialize LLM
    llm = ChatOpenAI(model="tinyllama-1.1b-chat-v1.0", temperature=0.4)

    # Step 4: Prompt
    template = """
    You are an intelligent assistant for answering questions from a PDF.
    Use the provided context to answer clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # Step 5: Question input
    st.markdown("---")
    user_question = st.text_input("💬 Ask your question here:")

    if user_question:
        with st.spinner("🤔 Thinking..."):
            retrieved_docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            formatted_prompt = prompt.format(context=context, question=user_question)
            response = llm.invoke(formatted_prompt)
            st.markdown("### 🧠 Answer:")
            st.write(response.content)

            with st.expander("📚 View context used"):
                st.text(context)
