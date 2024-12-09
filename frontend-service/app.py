import streamlit as st
import httpx
from typing import Optional

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {'AWS': [], 'GCP': []}

# Configure httpx client with longer timeout
client = httpx.Client(timeout=60.0)  # 60 second timeout

st.title("Cloud Documentation Q&A")

# Knowledge base selector
kb_type = st.sidebar.radio("Select Knowledge Base", ["AWS", "GCP"])

# File upload section
uploaded_file = st.sidebar.file_uploader(f"Upload {kb_type} Documentation", type=['pdf'])

if uploaded_file:
    try:
        # Send file to document service with kb_type
        files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
        response = client.post(
            f'http://document-service:8001/upload/{kb_type}', 
            files=files
        )
        if response.status_code == 200:
            st.sidebar.success(f"Successfully uploaded to {kb_type}")
        else:
            st.sidebar.error("Upload failed")
    except Exception as e:
        st.sidebar.error(f"Upload error: {str(e)}")

# Query section
query = st.text_input("Ask a question")

if query:
    try:
        # Send query to query service with longer timeout
        response = client.post(
            'http://query-service:8005/query',
            json={
                'question': query,
                'kb_type': kb_type,
                'chat_history': st.session_state.chat_history[kb_type]
            },
            timeout=120.0  # 2 minute timeout for queries
        )
        if response.status_code == 200:
            answer = response.json()['response']
            st.write("Answer:", answer)
            
            # Update chat history
            st.session_state.chat_history[kb_type].append(
                (query, answer)
            )
    except Exception as e:
        st.error(f"Query error: {str(e)}")
