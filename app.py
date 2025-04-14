import streamlit as st
import os
import base64
import tempfile
from io import BytesIO
from gtts import gTTS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
# Set page configuration
st.set_page_config(page_title="MediScan - Prescription Chatbot", layout="wide")

# Add title and description
st.title("MediScan - Prescription Chatbot")
st.markdown("Upload your prescription PDF and ask questions about it.")

# Set Google API key directly
os.environ["GOOGLE_API_KEY"] = "AIzaSyBlG32Owf2RddScsLPhktojqapIb8f2CQ4"

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Prescription PDF", type="pdf")

# Initialize session state for chat history and database
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for processed PDF database
if "pdf_db" not in st.session_state:
    st.session_state.pdf_db = None

# Function to convert text to speech and create an audio player
def text_to_speech(text, language="en"):
    try:
        # Map detected language to gTTS language code
        lang_map = {
            "English": "en",
            "Hindi": "hi",
            "Telugu": "te"
        }
        lang_code = lang_map.get(language, "en")
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to BytesIO object
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Encode to base64 for HTML audio player
        audio_base64 = base64.b64encode(audio_bytes.read()).decode()
        
        # Create HTML audio player with controls and autoplay
        audio_html = f'''
        <audio controls autoplay style="height: 30px; width: 100%">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        '''
        
        return audio_html
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Function to process PDF and create vector database
def process_pdf(pdf_file):
    try:
        # Create BytesIO object from uploaded file
        pdf_bytes = BytesIO(pdf_file.getvalue())
        
        # Use PyPDFLoader with file-like object instead of file path
        from langchain.document_loaders import PyPDFDirectoryLoader
        from langchain.document_loaders.pdf import BasePDFLoader
        
        class BytesIOPDFLoader(BasePDFLoader):
            """Load PDF files using BytesIO objects."""
            
            def __init__(self, file_content):
                self.file_content = file_content
                
            def load(self):
                import pypdf
                from langchain.docstore.document import Document
                
                pdf_reader = pypdf.PdfReader(self.file_content)
                return [
                    Document(
                        page_content=page.extract_text(),
                        metadata={"page": i},
                    )
                    for i, page in enumerate(pdf_reader.pages)
                ]
        
        # Use custom loader with BytesIO
        loader = BytesIOPDFLoader(pdf_bytes)
        pages = loader.load()
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(pages, embeddings)
        
        return db
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# Function to detect language
def detect_language(text):
    # Telugu Unicode range: U+0C00 to U+0C7F
    is_telugu = any('\u0c00' <= c <= '\u0c7f' for c in text)
    # Hindi Unicode range: U+0900 to U+097F
    is_hindi = any('\u0900' <= c <= '\u097f' for c in text)
    
    if is_telugu:
        return "Telugu"
    elif is_hindi:
        return "Hindi"
    else:
        return "English"

# Function to get response from Gemini
def get_response(db, query):
    try:
        docs = db.similarity_search(query)
        content = "\n".join([x.page_content for x in docs])
        
        # Detect language of the query
        language = detect_language(query)
        
        # Add language instruction to the prompt
        language_instruction = f"Please respond in {language}." if language != "English" else ""
        
        qa_prompt = f"""You are MediScan, a helpful and friendly medical prescription assistant. 
        
1. If the user asks a greeting like 'hello', 'how are you', or makes small talk, respond in a friendly, conversational manner.

2. For questions about the prescription or medical information:
   - Use the following pieces of context to answer accurately
   - If you don't know the answer, just say that you don't know, don't try to make up an answer
   - Explain medical terms in simple language

3. {language_instruction}

----------------"""
        input_text = qa_prompt+"\nContext:"+content+"\nUser question:\n"+query
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        result = llm.invoke(input_text)
        return result.content
    except Exception as e:
        st.error(f"Error getting response: {e}")
        return None

# Main app logic
if uploaded_file:
    # Process the uploaded PDF only if it hasn't been processed yet or if a new file is uploaded
    file_name = uploaded_file.name
    
    # Check if we need to process the PDF (new file or first time)
    if "current_file" not in st.session_state or st.session_state.current_file != file_name or st.session_state.pdf_db is None:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_db = process_pdf(uploaded_file)
            st.session_state.current_file = file_name
    
    if st.session_state.pdf_db:
        st.success("PDF processed successfully! You can now ask questions about the prescription.")
        
        # Chat interface
        st.subheader("Chat with your Prescription")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Add text-to-speech button for assistant messages
                if message["role"] == "assistant":
                    # Detect language of the response
                    detected_lang = detect_language(message["content"])
                    
                    # Create a button with speaker icon
                    col1, col2 = st.columns([10, 1])
                    with col2:
                        # Use a more unique key by combining index with a hash of the content
                        unique_key = f"tts_hist_{len(st.session_state.chat_history)}_{hash(message['content'])}"
                        if st.button("🔊", key=unique_key):
                            # Generate audio player
                            audio_html = text_to_speech(message["content"], detected_lang)
                            if audio_html:
                                # Display audio player
                                st.markdown(audio_html, unsafe_allow_html=True)
        
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
                    response = get_response(st.session_state.pdf_db, user_query)
                    if response:
                        st.write(response)
                        
                        # Detect language of the response
                        detected_lang = detect_language(response)
                        
                        # Add text-to-speech button
                        col1, col2 = st.columns([10, 1])
                        with col2:
                            # Use a unique key for the new response button
                            unique_key = f"tts_new_{len(st.session_state.chat_history)}_{hash(response)}"
                            if st.button("🔊", key=unique_key):
                                # Generate audio player
                                audio_html = text_to_speech(response, detected_lang)
                                if audio_html:
                                    # Display audio player
                                    st.markdown(audio_html, unsafe_allow_html=True)
                        
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
