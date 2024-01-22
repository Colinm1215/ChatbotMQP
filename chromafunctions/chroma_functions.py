from chromadb import *
import chromadb
from chromadb.utils import embedding_functions
from langchain.embeddings.openai import OpenAIEmbeddings
import config

openai_embeddings = OpenAIEmbeddings(openai_api_key = config.openai_api_key)
client = chromadb.PersistentClient(path='chroma_db4')

collection = client.create_collection(name="test", embedding_function=openai_embeddings)
# To get a collection
collection = client.get_collection(name="test", embedding_function=openai_embeddings)

docs = client.get_collection('docs_collection')

collection.add(
    documents=["lorem ipsum...", "doc2", "doc3", ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
langchain = client.get_collection('langchain')

collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)

collection.update(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...]
)

collection.delete(
    ids=["id1", "id2", "id3", ...],
    where={"chapter": "20"}
)
#collection = client.

print(docs.peek())
print(langchain.count())
