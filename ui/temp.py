import sys
import os

# Add the parent directory of the RAG module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from RAG import chunk_text, create_client, create_embeddings, create_vector_store, get_HF_Token, get_llm, get_model, get_pipeline, get_retriveQA, process_response, upload_pdf, write_response
import streamlit as st
import time
import os






## Backend code

def get_response(str_ques):

    pdf_files = [f for f in os.listdir("../KB/") if f.endswith('.pdf')]
    all_text=[]
    for pdf_file in pdf_files:
        pdf_path = os.path.join("../KB/", pdf_file)
        text = upload_pdf(pdf_path)
        if text:
            text = upload_pdf(pdf_path)
            all_text.extend(text)
            chunked=chunk_text(text)
            print(f"PDF {pdf_file} indexed successfully.")
        else:
            print(f"Failed to process PDF {pdf_file}.")
                
                  
    client=create_client()
    embeddings=create_embeddings()
    
    vector_database=create_vector_store(client, embeddings, all_text)

    token=get_HF_Token()
    model, tokenizer= get_model(token)
    
    
    pipeline_obj=get_pipeline(model, tokenizer)
    llm=get_llm(pipeline_obj)
    
    retrival_chain=get_retriveQA(llm, vector_database)
    
    truncated_response=[]
    #query=get_query(args)

    #for i in range(len(query)):
    response=get_response(retrival_chain, str_ques, vector_database)
    process_response(response, truncated_response)

    return truncated_response

def main():
    question="Who is Mr. Whisker?"
    
    ans=get_response(question)
    
    print(ans)
    
if __name__ == "__main__":
    main()