import streamlit as st
import os
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
    model_name = "meta-llama/Llama-2-7b-chat-hf"
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
    chunked = chunk_text(all_text)

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


## Backend code

def get_response(str_ques):
    pdf_files = [f for f in os.listdir("KB/") if f.endswith('.pdf')]
    all_text = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join("KB/", pdf_file)
        text = upload_pdf(pdf_path)
        if text:
            text = upload_pdf(pdf_path)
            all_text.extend(text)
            chunked = chunk_text(text)
            print(f"PDF {pdf_file} indexed successfully.")
        else:
            print(f"Failed to process PDF {pdf_file}.")

    client = create_client()
    embeddings = create_embeddings()

    vector_database = create_vector_store(client, embeddings, all_text)

    token = get_HF_Token()
    model, tokenizer = get_model(token)

    pipeline_obj = get_pipeline(model, tokenizer)
    llm = get_llm(pipeline_obj)

    retrival_chain = get_retriveQA(llm, vector_database)

    truncated_response = []
    # query=get_query(args)

    # for i in range(len(query)):
    response = get_response(retrival_chain, str_ques, vector_database)
    process_response(response, truncated_response)

    return truncated_response


## Backend code


## Frontend started!
st.title('RAG for PDF Text Retrieval')


def delete_file(filename):
    try:
        os.remove(os.path.join('KB', filename))
        return True
    except Exception as e:
        return False


def list_files():
    if os.path.exists('KB'):
        return os.listdir('KB')
    return []


files = list_files()
if files:
    st.subheader("PDF Files in the KB Folder are:")
    for index, file in enumerate(files, start=1):
        col1, col2 = st.columns([3, 1])
        col1.write(f"{index}. {file}")
        if col2.button(f"Delete {file}"):
            if delete_file(file):
                st.success(f"Deleted {file}")
                st.experimental_rerun()
            else:
                st.error("Failed to delete file")


def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('KB'):
            os.makedirs('KB')
        with open(os.path.join('KB', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        return False


def reset_form():
    st.session_state['question'] = ""


if 'question' not in st.session_state:
    st.session_state['question'] = ""

st.subheader("Upload Your PDFs Here")
uploaded_files = st.file_uploader("Choose PDF files:", type=['pdf'], accept_multiple_files=True, key="uploaded_files")

if st.button('Upload PDF'):
    if uploaded_files:
        success = [save_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]
        if all(success):
            st.success("All files have been uploaded successfully.")
            st.experimental_rerun()  # Refresh to update the file list immediately
        else:
            st.error("Some files were not uploaded successfully.")
    else:
        st.error("No files selected.")

st.subheader("Enter Your Query Here:")
question = st.text_input("Your questions:", value=st.session_state['question'], key='question')

if st.button('Get Answer'):
    # with st.spinner('Processing your request...'):
    #     time.sleep(2)

    st.subheader("Here is the Answer of Your Query:")
    answer = get_response(question)
    st.text_area("Answer", answer, height=300)

st.write("")
st.write("")

if st.button('Reset', on_click=reset_form):
    st.info('Please manually clear the file uploader by clicking the "X" beside each file.')

st.markdown("""
<style>
.stButton>button {
  float: right;
}
</style>
""", unsafe_allow_html=True)
