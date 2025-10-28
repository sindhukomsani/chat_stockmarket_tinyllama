import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ✅ Step 1: Setup LM Studio local model endpoint
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

# ✅ Step 2: Load PDF
pdf_path = "Stock_Market_Report.pdf"  # your file in tinyllama_chat folder
loader = PDFPlumberLoader(pdf_path)
docs = loader.load()
print(f"✅ Loaded {len(docs)} PDF pages")

# ✅ Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"✅ Split into {len(chunks)} chunks")

# ✅ Step 4: Create vector embeddings using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ Step 5: Define prompt
template = """You are an intelligent assistant for answering questions based on the given Stock Market Report.
Use the provided context to give concise and accurate answers.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate.from_template(template)

# ✅ Step 6: Initialize LLM (TinyLlama via LM Studio)
llm = ChatOpenAI(model="tinyllama-1.1b-chat-v1.0", temperature=0.3)

# ✅ Step 7: Build QA chain manually
def answer_question(query):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content

# ✅ Step 8: Ask something
if __name__ == "__main__":
    print("✅ System Ready! Ask questions about your Stock Market Report.")
    while True:
        query = input("\n💬 Question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = answer_question(query)
        print(f"\n🧠 Answer: {answer}\n")
