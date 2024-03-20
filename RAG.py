import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

def Chunk_text(text):
    text_splitter=CharacterTextSplitter(chunk_size=50, chunk_overlap=50)
    chunks=text_splitter.split_documents(text)
    
    return chunks

def storing_chunk(chunks):
    client = weaviate.Client(
    embedded_options = EmbeddedOptions()
    )

    vectorstore = Weaviate.from_documents(
        client = client,    
        documents = chunks,
        embedding = OpenAIEmbeddings(),
        by_text = False
    )
    return vectorstore

def upload_pdf(pdf_file):
    try:
        # Extract text from the PDF file
        text_loader = PyPDFLoader(pdf_file)  # Specify the encoding
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
    
    print(len(vectorstore))
