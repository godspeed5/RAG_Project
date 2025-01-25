import requests
import streamlit as st

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000"  # Update this with the backend server URL if deployed elsewhere

# Streamlit App Configuration
st.set_page_config(page_title="Chat with S3 Documents", page_icon="📃", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Application Title
st.title("Chat with All Documents in S3")
st.write("Ask questions and get answers with citations from your S3 documents.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for queries
user_input = st.chat_input("Type your question here...")
if user_input:
    # Display user's input immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare payload for the backend
    payload = {"user_input": user_input}
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id

    # Send request to the backend
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            assistant_response = data["response"]["answer"]
            st.session_state.session_id = data["session_id"]

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
