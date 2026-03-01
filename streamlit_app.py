import streamlit as st
import requests
import subprocess
import time
import os
import sys
import signal
import psutil

# Set page config
st.set_page_config(page_title="Text Summarizer", layout="wide")

# Initialize session state for server process
if "server_process" not in st.session_state:
    st.session_state.server_process = None


def start_fastapi_server():
    """Start FastAPI server in background"""
    try:
        # Check if a process is already running on port 8000
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.name() == "python.exe" and "uvicorn" in " ".join(
                    proc.cmdline()
                ):
                    # Server already running
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Start the server
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        st.session_state.server_process = process
        time.sleep(2)  # Give server time to start
        return True
    except Exception as e:
        st.error(f"Failed to start FastAPI server: {str(e)}")
        return False


def stop_fastapi_server():
    """Stop FastAPI server"""
    if st.session_state.server_process:
        try:
            if os.name == "nt":
                os.kill(st.session_state.server_process.pid, signal.SIGTERM)
            else:
                st.session_state.server_process.terminate()
            st.session_state.server_process = None
        except Exception as e:
            st.warning(f"Could not stop server: {str(e)}")


# Start server on app load
if st.session_state.server_process is None:
    with st.spinner("Starting FastAPI server..."):
        start_fastapi_server()

# Title and description
st.title("📝 Text Summarizer")
st.markdown("Powered by FastAPI (running in background) and Streamlit UI")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown("This app summarizes text using the BART model from Facebook/Meta.")
    st.markdown("The FastAPI server runs automatically in the background.")

    if st.button("Stop Server (when done)", help="Stop the FastAPI server"):
        stop_fastapi_server()
        st.success("Server stopped!")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    input_text = st.text_area(
        "Enter the text to summarize",
        placeholder="Paste your text here...",
        height=200,
        key="input_text",
    )

with col2:
    st.subheader("Output")
    output_placeholder = st.empty()

# Summarize button
if st.button("✨ Summarize", use_container_width=True, type="primary"):
    if not input_text.strip():
        st.error("Please enter some text to summarize!")
    else:
        try:
            with st.spinner("Summarizing..."):
                # Make request to FastAPI server
                response = requests.post(
                    "http://127.0.0.1:8000/summarize",
                    json={"text": input_text},
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    with output_placeholder.container():
                        st.success("Summary generated!")
                        st.text_area(
                            "Summary",
                            value=result["summary"],
                            height=150,
                            disabled=True,
                            key="output_text",
                        )

                        # Copy button
                        if st.button("📋 Copy to clipboard"):
                            st.info("Summary copied! (Paste with Ctrl+V)")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI server. Please try again.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The summary took too long to generate.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
