import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

def get_HF_Token():
    return os.environ.get('HF_TOKEN')

def get_model(token):
    model_name = "psmathur/orca_mini_3b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=True)
    tokenizer.bos_token_id = 1
    
    return model, tokenizer

def get_llm(pipeline_obj):
    llm = HuggingFacePipeline(pipeline=pipeline_obj)
    return llm

def get_retriveQA(llm, vector_db):
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )
    return qa_chain 

def get_pipeline(model, tokenizer):
    pipeline_obj = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return pipeline_obj

def get_response(retriveQA, query, vector_db):
    response = retriveQA.run(query)
    return response

def upload_pdf(pdf_file):
    try:
        text_loader = PyPDFLoader(pdf_file)
        text = text_loader.load_and_split()
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None
    return text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)
    split_docs = text_splitter.split_documents(text)
    return split_docs

def create_client():
    WEAVIATE_URL = "https://coms-579-c9s1i04n.weaviate.network"

    client = weaviate.Client(
        url=WEAVIATE_URL)
    return client

def create_embeddings():
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name, 
    model_kwargs=model_kwargs
    )
    return embeddings

def create_vector_store(client, embeddings, all_text):
    chunked=chunk_text(all_text)

    vector_db = Weaviate.from_documents(
        chunked, embeddings, client=client, by_text=False
    )
    return vector_db

def process_response(response, all_response):
    response_lines = response.split("\n")

    for i in range(len(response_lines)):
        if response_lines[i].startswith('Question:'):
            all_response.append(response_lines[i])
        elif response_lines[i].startswith('Helpful Answer:'):
            all_response.append(response_lines[i])
            
'''def get_query(args):
    txt_files = [f for f in os.listdir(args.query_folder) if f.endswith('.txt')]
    
    for txt in txt_files:
        query_path = os.path.join(args.query_folder, txt)
        with open(query_path) as file:
            query=file.readlines()
    return query'''

    
            
def write_response(truncated_response):
    response_file_path = "Response/response.txt"
    os.makedirs(os.path.dirname(response_file_path), exist_ok=True)

    with open(response_file_path, "a") as file:
        for line in truncated_response:
            file.write(line + "\n")
            

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Upload and index PDF files")
#     parser.add_argument("--pdf_folder", type=str, help="Path to the folder containing PDF files")
#     parser.add_argument("--query_folder", type=str, help="Path to Query Folder")
#     args = parser.parse_args()
#     question=""

#     if args.pdf_folder:
#         pdf_files = [f for f in os.listdir(args.pdf_folder) if f.endswith('.pdf')]
#         all_text=[]
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(args.pdf_folder, pdf_file)
#             text = upload_pdf(pdf_path)
#             if text:
#                 text = upload_pdf(pdf_path)
#                 all_text.extend(text)
#                 chunked=chunk_text(text)
#                 print(f"PDF {pdf_file} indexed successfully.")
#             else:
#                 print(f"Failed to process PDF {pdf_file}.")
#     else:
#         print("Please provide the path to the folder containing PDF files using --pdf_folder argument.")
        
#     client=create_client()
#     embeddings=create_embeddings()
    
#     vector_database=create_vector_store(client, embeddings, all_text)

#     token=get_HF_Token()
#     model, tokenizer= get_model(token)
    
    
#     pipeline_obj=get_pipeline(model, tokenizer)
#     llm=get_llm(pipeline_obj)
    
#     retrival_chain=get_retriveQA(llm, vector_database)
    
#     truncated_response=[]
#     #query=get_query(args)

#     #for i in range(len(query)):
#     response=get_response(retrival_chain, question, vector_database)
#     process_response(response, truncated_response)

#     write_response(truncated_response)
    
    
    
    
    
    
