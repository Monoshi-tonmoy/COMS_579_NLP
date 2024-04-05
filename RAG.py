import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def query_model(query_text, vector_store):
    #todo: We have to query LLM models on the basis of our weaviate vector
    
    '''# Encode the query text into embeddings using a Hugging Face model
    model_name = "sentence-transformers/all-MiniLM-l6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        query_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

    weaviate_embeddings = vector_store._embedding

    similarities = cosine_similarity([query_embeddings], weaviate_embeddings)

    most_similar_index = similarities.argmax()

    most_similar_document = vector_store.get_object(most_similar_index)

    return most_similar_document'''



def chunk_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_documents(text)
    return chunks

def store_chunks(chunks, client):
    model_path = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_store = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        by_text=True
    )
    return vector_store

def upload_pdf(pdf_file):
    try:
        text_loader = PyPDFLoader(pdf_file)
        text = text_loader.load()
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload and index PDF files")
    parser.add_argument("--pdf_folder", type=str, help="Path to the folder containing PDF files")
    args = parser.parse_args()

    if args.pdf_folder:
        pdf_files = [f for f in os.listdir(args.pdf_folder) if f.endswith('.pdf')]
        client = weaviate.Client(embedded_options=EmbeddedOptions())
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.pdf_folder, pdf_file)
            text = upload_pdf(pdf_path)
            if text:
                chunked_text = chunk_text(text)
                vector_store = store_chunks(chunked_text, client)
                print(f"PDF {pdf_file} indexed successfully.")
            else:
                print(f"Failed to process PDF {pdf_file}.")
    else:
        print("Please provide the path to the folder containing PDF files using --pdf_folder argument.")

    query_text="Who lived in the small village of rolling hills?"
    result = query_model(query_text, vector_store)
    print(result)