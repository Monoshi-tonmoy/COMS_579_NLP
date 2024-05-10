# RAG for PDF Text Retrieval

## Overview
This Python script provides a system for uploading, indexing, and querying text from PDF files. It utilizes various natural language processing (NLP) and machine learning (ML) techniques for text processing and retrieval.

## Features
- **PDF Upload:** Upload PDF files for text extraction and indexing.
- **Text Indexing:** Index the text extracted from PDF files for efficient retrieval.
- **Querying:** Query the indexed text to retrieve relevant information.
- **Hugging Face Integration:** Utilizes Hugging Face models for text generation and question answering.
- **Weaviate Integration:** Integrates with Weaviate for vector storage and retrieval.

## Prerequisites
- Python 3.x
- Hugging Face API token (exported as an environment variable)
- Hugging Face account with the CLI installed and logged in
- Weaviate instance URL

## Installation
1. Clone this repository to your local machine:

    ```
    git clone https://github.com/Monoshi-tonmoy/COMS_579_NLP.git
    ```

2. Install the required Python packages:

    ```
    pip install -r requirements.txt

    ```

3. Export your Hugging Face API token as an environment variable:

    ```
    export HF_TOKEN=your_hugging_face_token
    ```

## Usage

### To run our project in the terminal, do the following steps:
1. Place your PDF files in the KB folder.
2. Run the script with the `--pdf_folder` argument pointing to the folder containing the PDF files and the `--query_folder` argument pointing to the folder containing query files.

    ```
    python RAG.py --pdf_folder KB --query_folder Query
    ```

3. The script will extract text from the PDF files, index them, and perform queries based on the queries provided in the query files.
4. The responses to the queries will be saved in a `Response` folder as `response.txt`.


### To run our project in the GUI, do the following steps:

1. Open the console and goto Project Root
2. Make sure you choose the appropriate environment (installing from requirements.txt)
3. Run this command: **streamlit run ui/main_console.py**
4. You can see the use interface. 
5. You can upload, edit or delete pdf; can write question & get answer; and also can reset everything using our GUI implemented using **streamlit** library.
6. If you find hard to understand, follow the video tutorial.
7. Thanks for your patience.

## Notes
- Make sure you login to huggingface using your token.
- Ensure that your Hugging Face API token is properly exported before running the script.
- Modify the `WEAVIATE_URL` variable in the script to match your Weaviate instance URL.

Here find the videos of the milestones [https://drive.google.com/drive/folders/1KgwPmULG0o5s9hEQSHJrN_bqGFbt2mlu?usp=drive_link] drive.


