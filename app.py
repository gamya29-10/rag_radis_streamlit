import streamlit as st
from langchain_redis import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# UI
st.title("Redis + LLM Demo")

# Redis
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embeddings=embeddings
)

set_llm_cache(cache)

# ✅ LOAD PROPER QA MODEL
model_name = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Input
query = st.text_input("Ask your question:")

# Simple knowledge base (can expand later)
context = """
France is a country in Europe. Its capital is Paris.
Germany's capital is Berlin.
India's capital is New Delhi.
"""

if query:
    inputs = tokenizer(query, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )

    st.success(answer)