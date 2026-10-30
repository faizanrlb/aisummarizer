
import streamlit as st
import re
from transformers import pipeline

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("Faizan Rasheed | ID: B02-0923-000031")
st.subheader("AI Text Summarizer")
st.write("Summarize your text using artificial intelligence.")

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


# LOAD MODEL ONLY ONCE
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model=MODEL_NAME,
        device=-1
    )


# WORD COUNTER
def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


# TEXT SUMMARIZATION
def generate_summary(text, summarizer, max_length, min_length):

    tokenizer = summarizer.tokenizer

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    # Split long text into chunks
    chunk_size = 850
    summaries = []

    for i in range(0, len(tokens), chunk_size):

        chunk_tokens = tokens[i:i + chunk_size]

        chunk = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        actual_max = min(
            max_length,
            max(20, int(len(chunk_tokens) * 0.8))
        )

        actual_min = min(
            min_length,
            max(8, int(actual_max * 0.3))
        )

        if actual_min >= actual_max:
            actual_min = max(5, actual_max - 5)

        result = summarizer(
            chunk,
            max_length=actual_max,
            min_length=actual_min,
            do_sample=False,
            truncation=True
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


# CLEAR INPUT CALLBACK
def clear_input():
    st.session_state["input_text"] = ""
    st.session_state.pop("summary", None)


# SIDEBAR
st.sidebar.header("Summary Settings")

length_option = st.sidebar.selectbox(
    "Choose Summary Length",
    ["Short", "Balanced", "Detailed"]
)

settings = {
    "Short": (80, 20),
    "Balanced": (120, 30),
    "Detailed": (160, 40)
}

max_len, min_len = settings[length_option]


# USER INPUT
user_text = st.text_area(
    "Enter your text below:",
    height=250,
    key="input_text",
    placeholder="Paste your article or paragraph here..."
)

original_count = count_words(user_text)

st.info(f"Original Word Count: {original_count}")

col1, col2 = st.columns(2)

with col1:
    summarize = st.button(
        "Generate Summary",
        type="primary",
        use_container_width=True
    )

with col2:
    st.button(
        "Clear Text",
        on_click=clear_input,
        use_container_width=True
    )


# GENERATE SUMMARY
if summarize:

    if not user_text.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner("Loading AI model..."):

                summarizer = load_model()

            with st.spinner("Generating summary..."):

                summary = generate_summary(
                    user_text,
                    summarizer,
                    max_len,
                    min_len
                )

            st.session_state["summary"] = summary

        except Exception as e:
            st.error(f"Error: {e}")


# DISPLAY RESULTS
if st.session_state.get("summary"):

    summary = st.session_state["summary"]

    summary_count = count_words(summary)

    reduction = (
        (original_count - summary_count)
        / original_count * 100
        if original_count > 0 else 0
    )

    st.divider()

    st.header("Generated Summary")

    st.write(summary)

    st.subheader("Word Count Comparison")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Words",
        original_count
    )

    c2.metric(
        "Summary Words",
        summary_count
    )

    c3.metric(
        "Reduction",
        f"{reduction:.1f}%"
    )

    st.download_button(
        "Download Summary",
        data=summary,
        file_name="summary.txt",
        mime="text/plain"
    )

st.divider()

st.caption(
    "Developed by Faizan Rasheed | ID: B02-0923-000031"
)

