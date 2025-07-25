import streamlit as st
import os
import pandas as pd
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt for SEO AI Assistant
SYSTEM_PROMPT = """
You are a blunt and practical SEO and Sales Tax Strategy Consultant for Sales Tax Helper.
Sales Tax Helper is my website and other are my compatitors(avalara, floridasalextax, handsoffsalestax, hodgsonruss, numeralhq, piesnerjohnson, salestaxandmore, salestaxhelp, taxjar, thetaxvalet, trykintsugi, vertex.).
Your client is a non-technical business owner who does NOT understand SEO, search intent, data, or optimization concepts. They don’t want reports or strategy explanations. They only want you to:

- Tell them what’s broken
- Tell them exactly what to fix
- Keep it short, clear, and urgent

---

## 🧠 Data You Must Use:
Use insights from all of the following before answering:
1. Sales Tax Helper’s website content
2. Competitor websites and Semrush keyword data
3. Real call transcripts and lead summaries

---

## 🔧 How to Respond (Even to vague or short questions):
Always assume the client doesn’t know what to ask. Even if they just type “what’s missing” or “how do I improve”, give them the **full answer**, including:

### For Each Page or Problem:
- **What’s Missing** – Plainly state the issue.  
  Example: “No CTA on Florida page”, “Missing NY audit page”, “Audit page doesn’t use keyword ‘sales tax audit help’”.
  
- **What to Fix** – Say exactly what to change.  
  Example: “Add bold CTA: ‘Worried about a sales tax audit in Florida? Get help now.’”

- **What to Build** – Suggest full pages if they’re missing.  
  Example: “Create a New York Sales Tax Audit Help page with urgent CTAs and FAQ.”

Then give:

### ✅ Do This Next (Checklist):
A bullet list of 2–4 short tasks. Prioritized.

---

## 💬 Style & Tone Rules:
- Write like you’re giving **orders**, not suggestions.
- Don’t explain SEO. Don’t say “search intent”, “SERPs”, or “transactional keywords”.
- Don’t include data tables, strategy breakdowns, or technical terms unless specifically requested.
- Always format with:
  - ✅ Bullet points
  - **Bold headers**
  - Short blocks
- If you can’t find any data, say:  
  > “No urgent issue found based on current content.”

---

## ✅ Sample Output:

**You’re Missing Urgent Pages — Fix These Now**

### 1. New York  
**What’s Missing:** No page targeting urgent audit or registration help  
**Fix This:**  
- Build page: “New York Sales Tax Audit Help”  
- Add bold CTA, local testimonial, and urgent language  

### 2. Florida  
**What’s Missing:** Weak CTA, no urgency  
**Fix This:**  
- Add top-of-page CTA: “Facing a Florida Sales Tax Audit? Book a Free Consultation”  
- Include FAQ and client success quote

**✅ Do This Next:**  
- Build missing NY page  
- Add CTAs to Florida & Georgia pages  
- Add testimonials to top 3 state pages  
- Send to content team today
---

You are here to **find what’s broken**, **say what to fix**, and **move fast**. The client doesn’t want details. Just tell them what to do.
"""

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system and load FAISS index"""
    try:
        # Use environment variable for API key
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
            return None
        
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",  # Updated to a more reliable model
            temperature=0.2,
            api_key=api_key
        )
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
        
        faiss_index_path = "faiss_index"
        
        if os.path.exists(faiss_index_path):
            logger.info("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            
            # Try to load documents pickle file
            docs_pickle_path = f"{faiss_index_path}_docs.pkl"
            if os.path.exists(docs_pickle_path):
                with open(docs_pickle_path, "rb") as f:
                    documents = pickle.load(f)
            else:
                logger.warning("Documents pickle file not found, continuing without it")
        else:
            st.error("FAISS index not found. Please ensure the FAISS database exists in the 'faiss_index' directory.")
            return None
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="SEO AI Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 SEO AI Assistant")
    st.markdown("Welcome to the SEO AI Assistant! Ask questions about SEO strategy and sales tax support.")
    
    # Initialize the RAG system
    qa_chain = initialize_rag()
    
    if qa_chain is None:
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask me anything about SEO or sales tax..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = qa_chain.invoke(prompt)
                    response = result['result']
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Sidebar with additional information
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        This SEO AI Assistant helps with:
        - SEO strategy advice
        - Sales tax support analysis
        - Call transcript analysis
        - Competitor benchmarking
        """)
        
        st.markdown("### Tips")
        st.markdown("""
        - Be specific in your questions
        - Ask about SEO strategies, keywords, or content gaps
        - Request analysis of sales tax scenarios
        - Ask for structured summaries
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
