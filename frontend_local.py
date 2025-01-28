import requests
import streamlit as st

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Streamlit configuration
st.set_page_config(page_title="Chat with Documents", page_icon="📁", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.title("Chat with Local Documents")
st.write("Upload PDF files and ask questions.")

# Upload files
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose PDF files to upload", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Uploading {file.name}...")
        try:
            response = requests.post(f"{BACKEND_URL}/upload", files={"file": (file.name, file.getvalue())})
            if response.status_code == 200:
                st.success(f"{file.name} uploaded successfully!")
            else:
                st.error(f"Failed to upload {file.name}: {response.text}")
        except Exception as e:
            st.error(f"Error uploading {file.name}: {e}")

# Search bar
st.header("Search Documents")
query = st.text_input("Ask a question about the uploaded documents")

if query:
    payload = {"user_input": query, "session_id": st.session_state.session_id}
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state.session_id = data["session_id"]
            answer = data["response"]["answer"]
            st.markdown("### Response:")
            st.write(answer)
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to the backend: {e}")
