import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import tempfile
# Set page configuration
st.set_page_config(page_title="MediScan - Prescription Chatbot", layout="wide")

# Add title and description
st.title("MediScan - Prescription Chatbot")
st.markdown("Upload your prescription PDF and ask questions about it.")

# Set Google API key directly
os.environ["GOOGLE_API_KEY"] = "AIzaSyBlG32Owf2RddScsLPhktojqapIb8f2CQ4"

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Prescription PDF", type="pdf")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to process PDF and create vector database
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(pages, embeddings)
        
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# Function to get response from Gemini
def get_response(db, query):
    try:
        docs = db.similarity_search(query)
        content = "\n".join([x.page_content for x in docs])
        qa_prompt = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.----------------"
        input_text = qa_prompt+"\nContext:"+content+"\nUser question:\n"+query
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        result = llm.invoke(input_text)
        return result.content
    except Exception as e:
        st.error(f"Error getting response: {e}")
        return None

# Main app logic
if uploaded_file:
    # Process the uploaded PDF
    with st.spinner("Processing PDF..."):
        db = process_pdf(uploaded_file)
    
    if db:
        st.success("PDF processed successfully! You can now ask questions about the prescription.")
        
        # Chat interface
        st.subheader("Chat with your Prescription")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Get user query
        user_query = st.chat_input("Ask a question about your prescription")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(db, user_query)
                    if response:
                        st.write(response)
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        st.error("Failed to get a response. Please try again.")
else:
    st.info("Please upload a prescription PDF file to get started.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About MediScan")
st.sidebar.markdown("MediScan helps you understand your medical prescriptions using AI.")
