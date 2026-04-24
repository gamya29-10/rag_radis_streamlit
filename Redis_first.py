from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisCache, RedisSemanticCache
from langchain_core.globals import set_llm_cache
import redis

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embeddings=embeddings,
    distance_threshold=0.2,
    ttl=120
))

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

prompt = ChatpromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = promt | llm

response = chain.invoke({"question": "What is the capital of France?"})
print(response.content)





