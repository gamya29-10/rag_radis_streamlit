from langchain_community.cache import RedisSemanticCache
from langchain_core.globals import set_llm_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

set_llm_cache(
    RedisSemanticCache(
        redis_url="redis://localhost:6379",
        embedding=embeddings
    )
)

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke({
    "question": "How do I fix a flat tire?"
})

print(response.content)