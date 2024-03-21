# RAG

This project aims to analyze documents, such as PDF files, using Weaviate for document vectorization and a Language Model (LLM) from Hugging Face for text generation.

## Setup

1. **Clone the repository:** `git clone <repository_url>`
2. **Install the required dependencies:** `pip install -r requirements.txt`

3. **Download pre-trained models:**
   - **Hugging Face LLM:** You can choose any desired LLM model from Hugging Face's model hub. Update the `modelPath` variable in `storing_chunk()` function in `main.py` with the desired model path.

4. **Run the script:** `python main.py --pdf_file <path_to_pdf>`


## Workflow

1. **Upload PDF**: Use the `upload_pdf()` function to load a PDF file and extract its text.

2. **Chunk Text**: Chunk the extracted text into smaller segments using the `Chunk_text()` function.

3. **Store Chunks**: Store the text chunks in Weaviate for vectorization using the `storing_chunk()` function.

4. **Query Model**: Implement the `query_model()` function to use the Weaviate vector store to query the LLM model for text generation.

## To-Do

1. **Implement Query Model Function**: Finish the `query_model()` function to utilize the Weaviate vector store for querying the LLM model. This involves retrieving embeddings from Weaviate and integrating them with the LLM for text generation.

