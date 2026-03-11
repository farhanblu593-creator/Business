import os
import json
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ----------------------- CONFIGURATION -----------------------

# 1. Models
# Using DeepSeek-R1 for both "thinking" and "answering"
Embedding_Model = OllamaEmbeddings(model="deepseek-r1:1.5b")
Language_Model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.1) # Lower temperature for facts

# 2. Storage
# This is the "In-Memory" brain - no FAISS files saved to disk
Document_Vector_db = InMemoryVectorStore(Embedding_Model)

# ----------------------- CORE FUNCTIONS -----------------------

def discover_business_structure(docs):
    """Step 1: Automatically discover what business and categories are in the PDF."""
    # Take a snippet of the first few pages
    sample_text = "\n---\n".join([d.page_content[:600] for d in docs[:3]])
    
    discovery_prompt = f"""
    You are a Business Intelligence Analyst. Analyze these snippets and identify:
    1. The Name of the Organization.
    2. 4 logical 'Service Categories' to group these documents.
    
    SNIPPETS:
    {sample_text}

    RETURN ONLY A JSON OBJECT:
    {{"business_name": "...", "categories": ["Cat1", "Cat2", "Cat3", "Cat4"]}}
    """
    
    response = Language_Model.invoke(discovery_prompt)
    # Clean the response for JSON parsing
    try:
        clean_json = response.split('{')[-1].split('}')[0]
        return json.loads("{" + clean_json + "}")
    except:
        return {"business_name": "Universal Business", "categories": ["General Services"]}

def generate_standardized_answer(query, context, business_name):
    """Step 4: Re-write messy PDF text into the clean 'Service Article' style."""
    PROMPT_TEMPLATE = f"""
    You are a Senior Consultant for {business_name}. 
    Answer the query using the provided context. 
    
    FOLLOW THIS FORMAT (STEP 4 STANDARD):
    1. Service Objective: (A simple 1-sentence summary)
    2. Eligibility: (Bullet points of who can apply)
    3. Business Rules: (Step-by-step instructions)
    
    If the information is not in the context, say you don't know.

    Query: {{user_query}}
    Context: {{document_context}}
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = prompt | Language_Model
    return response_chain.invoke({"user_query": query, "document_context": context})

# ----------------------- STREAMLIT UI -----------------------

st.set_page_config(page_title="Universal Knowledge App", layout="wide")
st.title("🚀 Universal Service Assistant")

uploaded_pdf = st.file_uploader("Upload any Business PDF", type="pdf")

if uploaded_pdf:
    # 1. Load & Chunk
    if "processed" not in st.session_state:
        with st.spinner("Reading and Analyzing Documents..."):
            # Load
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            loader = PDFPlumberLoader("temp.pdf")
            raw_docs = loader.load()
            
            # Step 1: Discover Business Identity
            st.session_state["business_map"] = discover_business_structure(raw_docs)
            
            # Step 2: Indexing
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(raw_docs)
            Document_Vector_db.add_documents(chunks)
            
            st.session_state["processed"] = True
            st.success(f"✅ Configured for {st.session_state['business_map']['business_name']}")

    # Display Sidebar Categories
    if "business_map" in st.session_state:
        st.sidebar.title(st.session_state["business_map"]["business_name"])
        st.sidebar.write("### Detected Departments:")
        for cat in st.session_state["business_map"]["categories"]:
            st.sidebar.info(cat)

    # Chat UI
    user_query = st.chat_input("Ask a question about the services...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.spinner("Formatting response..."):
            # Retrieve
            related_docs = Document_Vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in related_docs])
            
            # Generate (Standardized style)
            answer = generate_standardized_answer(
                user_query, 
                context, 
                st.session_state["business_map"]["business_name"]
            )
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(answer)