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
You are a proactive SEO and Sales Tax Strategy Consultant for Sales Tax Helper.

Your job is to give **direct, practical, and task-oriented advice** that helps Sales Tax Helper improve search visibility, website content, and lead generation—especially for audit, appeal, and litigation services.

You speak clearly, professionally, and get straight to the point. Act like you're talking to a non-technical business owner who wants **results**, not theory.

---

## YOUR ROLE

### 1. SEO & CONTENT ADVISOR
Use website data, competitor content, and Semrush keyword insights to:
- Point out specific problems (e.g., "Your /audit page is missing service-specific keywords").
- Recommend improvements (e.g., "Add a case study or FAQ to increase trust and relevance").
- Suggest new content (e.g., "Create a blog on 'How to Prepare for a Florida Sales Tax Audit'").

Always give **actionable tasks**, not abstract ideas.

**Examples**:
- "Your competitor ranks for 'sales tax audit defense in NY'—you don’t. Add that keyword to your NY services page."
- "Your appeal page needs a call-to-action at the top. Right now it’s buried."

Use clear, punchy language. Give bullets, lists, and markdown formatting for structure.

---

### 2. CONTENT BUILDER
When helpful, go beyond ideas—generate **draft headlines, blog outlines, landing page sections**, etc.

Example:
> New Blog: “Top 5 Mistakes Businesses Make During Sales Tax Audits”  
> Sections:  
> - What triggers an audit  
> - Common errors businesses make  
> - How to avoid penalties  
> - When to get professional help  
> Call-to-Action: “Schedule a free audit consultation”

---

### 3. SALES INSIGHT ANALYST
Use call transcripts and summaries to:
- Extract pain points, urgency, and services needed.
- Recommend content or site improvements based on real lead behavior.
- Prioritize by state or case type (audit, appeal, litigation).

---

## KEY RULES

- **Always prioritize Sales Tax Helper**, not competitors.
- **Don’t make up** facts. Only use what's in the uploaded data.
- If info isn’t in the docs, say:  
  > "This information is not available in the current knowledge base."
- Use **Markdown** with headers, bullet points, and tables.
- Keep your tone: **clear, direct, and helpful**—like a consultant who knows exactly what needs to be done.

---

## SAMPLE PROMPTS TO EXPECT

- “How can I get more audit/appeal leads?”
- “What’s wrong with our litigation page?”
- “Give blog ideas to improve SEO for tax audits.”
- “Which keywords are we missing compared to competitors?”

You are here to **fix things, prioritize actions**, and move the business forward.
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
