import os
import sys
from pathlib import Path

# # Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# from src.retrieval.vector_store import PineconeVectorStore

# store = PineconeVectorStore(
#     index_name=os.environ["PINECONE_INDEX_NAME"],
#     namespace="tier1_v1",
# )

# stats = store.index.describe_index_stats()
# print(stats.namespaces)

# test_id = "fine_tuning::section::chunk::3"

# # Fetch ANY 5 vectors from the namespace
# res = store.index.query(
#     vector=[0.0] * 3072,   # dummy vector, won't be used for similarity
#     top_k=5,
#     include_metadata=True,
#     namespace="tier1_v1",
# )

# print(res)
# for match in res.matches:
#     print("VECTOR ID:", match.id)
#     print("DOC ID:", match.metadata.get("doc_id"))
#     print("-" * 40)

# # res = store.index.fetch(
# #     ids=[test_id],
# #     namespace="tier1_v1",
# # )

# # print("Fetched IDs:", res.vectors.keys())

from openai import OpenAI
from src.retrieval.vector_store import PineconeVectorStore
import os

client = OpenAI()

store = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    namespace="tier1_v1",
)

# Use a REAL query
query = "fine-tuning methods in language models"

# Generate a real embedding
emb = client.embeddings.create(
    model="text-embedding-3-large",
    input=query,
).data[0].embedding

# Query Pinecone
res = store.index.query(
    vector=emb,
    top_k=5,
    include_metadata=True,
    namespace="tier1_v1",
)

for m in res.matches:
    print("VECTOR ID:", m.id)
    print("DOC ID:", m.metadata.get("doc_id"))
    print("SECTION:", m.metadata.get("section_path"))
    print("-" * 50)