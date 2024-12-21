import streamlit as st
from transformers import pipeline

# load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# streamlit app
st.title("Text Summarizer")
input_text = st.text_area("Enter the text", placeholder="Enter the text to summarize here...")

# summarize button
if st.button("Summarize"):
    if input_text.strip():
        with st.spinner("Summarizing..."):
            try:
                # generate summary
                summary = summarizer(input_text, max_length=70, min_length=35, do_sample=False)
                st.subheader("Summary")
                st.write(summary[0]["summary_text"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter some text to summarize!")