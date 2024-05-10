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

## Backend code
















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
