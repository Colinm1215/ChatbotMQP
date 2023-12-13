from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
#from chromadb.config import Settings
#settings=Settings(allow_reset=True, anonymized_telemetry=False)
import config

openai_embeddings = OpenAIEmbeddings(openai_api_key = config.openai_api_key)

def chunk_text_from_txt(txtpath, chunk_size, overlap):  # Ada 2's optimal chunk size
    loader = TextLoader(txtpath, encoding = 'UTF-8')
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = chunk_size, chunk_overlap = overlap, separators= '/f')
    extracted = loader.load_and_split(text_splitter)
    return extracted

def extract_and_chunk_text_from_pdf(pdfpath, chunk_size, overlap):  # Ada 2's optimal chunk size
    reader = PyPDFLoader(pdfpath)
    text_splitter = RecursiveCharacterTextSplitter (chunk_size = chunk_size, chunk_overlap = overlap)
    extracted = reader.load_and_split(text_splitter)
    # Chunking the text for openai ada embbed model
    #chunks = text_splitter.create_documents(extracted)
    return extracted

def embed_and_store_documents(data, model, collection): #embeds list of documents (class) and returns collection specified
    store_target = Chroma.from_documents(
        documents = data, 
        embedding = model, 
        collection_name = collection, 
        persist_directory = config.vectordb_path)
    return store_target

def process_txt(txtpath, collection, chunk_size = 500, overlap = 100, model = openai_embeddings):
    chunkedtxt = chunk_text_from_txt(txtpath, chunk_size, overlap)
    return embed_and_store_documents(chunkedtxt, model, collection)

def initialize_db():
    print("Doing first time setup for ChromaDB with WPI files...")
    #catalog = chunk_text_from_txt("./wpi_files/catalog.txt")
    #guide = chunk_text_from_txt("./wpi_files/GompeisGuide-2.txt")
    #extracted = loader.load_and_split(text_splitter)
    process_txt(txtpath = "./wpi_files/catalog.txt", collection = "wpi_docs")
    print("Loaded catalog...")
    process_txt(txtpath = "./wpi_files/GompeisGuide-2.txt", collection = "wpi_docs")
    print("Loaded guide...")
    config.runsetup = False

#vectordb = Chroma(persist_directory="chromadb", embedding_function=openai_embeddings, collection_name = 'wpi_docs')
#print(vectordb._collection.count())

#docs_collection = chroma_client.get('docs_collection')
#user_collection = chroma_client.get(userID)
#print(chroma_client.list_collections())
#print(user_collection.count())
#print(docs_collection.count())
#print(docs_collection.peek(1))
#Chroma(persist_directory="chroma_db4", embedding_function=openai_embeddings)
#textmoment2 = doc_collection.query(query)

#initialize_db()
#docs_collection = Chroma("wpi_docs", embedding_function=openai_embeddings, persist_directory = config.vectordb_path)
#user_collection = Chroma(config.userID, embedding_function=openai_embeddings, persist_directory = 'chromadb')
