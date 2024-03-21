import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM



def query_model(query_text, vector_store):
#todo: Finish the query model function using a small LLM model
    pass

def Chunk_text(text):
    text_splitter=CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks=text_splitter.split_documents(text)
    
    return chunks

def storing_chunk(chunks):
    client = weaviate.Client(
    embedded_options = EmbeddedOptions()
    )
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"


    model_kwargs = {'device':'cpu'}

    encode_kwargs = {'normalize_embeddings': False}


    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs 
    )
    vectorstore = Weaviate.from_documents(
        client = client,    
        documents = chunks,
        embedding = embeddings,
        by_text = True
    )
    return vectorstore


def upload_pdf(pdf_file):
    try:
        text_loader = PyPDFLoader(pdf_file) 
        text = text_loader.load()
    except Exception as e:
        print(f"Error uploading PDF: {e}")
    
    return text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload and index PDF files")
    parser.add_argument("--pdf_file", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    if args.pdf_file:
        text=upload_pdf(args.pdf_file)
    else:
        print("Please provide the path to the PDF file using --pdf_file argument.")
        
    #Chunking the Loaded text from our given pdfs
    chunked_text=Chunk_text(text)
    
    vectorstore=storing_chunk(chunked_text)
    
    #help(vectorstore)

    
    query_text = "How much dollars cost annually for tech giants since they fail to ensure the quality of softwares?"
    generated_text = query_model(query_text, vectorstore)
    print("Generated Text:", generated_text)