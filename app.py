import streamlit as st

st.set_page_config(page_title="AI Content Rewriter", layout="centered")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

@st.cache_resource
def load_rewriter():
    model_name = "t5-small"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    rewriter = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # force CPU
    )
    return rewriter

rewriter = load_rewriter()

st.title("AI Content Rewriter")
st.write("Rewrite any text in your preferred tone — fully local and private.")

text_input = st.text_area("Enter the text you want to rewrite:")
tone = st.selectbox(
    "Choose a tone:",
    ["Formal", "Friendly", "Persuasive", "Simple", "Professional"]
)

def rewrite_text(text, tone):
    prompt = f"Rewrite the following text in a {tone} tone:\n{text}"
    outputs = rewriter(prompt, max_length=256, num_beams=4)
    return outputs[0]['generated_text']

if st.button("Rewrite Text"):
    if not text_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Rewriting text locally..."):
            result = rewrite_text(text_input, tone)
        st.success("Rewritten Output:")
        st.write(result)

st.caption("Powered by Hugging Face Transformers (Offline Mode)")
