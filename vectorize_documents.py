import os
import time
import logging
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Loading the embedding models
embeddings = HuggingFaceEmbeddings()

# Define the directory and the path to the log of processed files
data_directory = "data"
processed_files_log = "processed_files.txt"

# Load already processed files
if os.path.exists(processed_files_log):
    with open(processed_files_log, 'r') as file:
        processed_files = set(file.read().splitlines())
else:
    processed_files = set()

# Initialize a list for new documents
new_documents = []

# Iterate over all files in the directory
for root, dirs, files in os.walk(data_directory):
    for file_name in files:
        if file_name.endswith(".pdf"):
            file_path = os.path.join(root, file_name)
            if file_path not in processed_files:
                start_time = time.time()
                logging.info(f"Processing new file: {file_path}")

                try:
                    # Load each PDF document individually with OCR
                    loader = UnstructuredFileLoader(file_path, strategy="ocr_only")
                    document = loader.load()

                    # Split document into chunks
                    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
                    text_chunks = text_splitter.split_documents(document)

                    if text_chunks:
                        # Add chunks to the new_documents list
                        new_documents.extend(text_chunks)

                        # Log processing time
                        elapsed_time = time.time() - start_time
                        logging.info(f"Processed {file_name} in {elapsed_time:.2f} seconds")
                    else:
                        logging.warning(f"PDF text extraction failed for {file_name}, skipping text extraction...")

                    # Add the file to the processed files set
                    processed_files.add(file_path)
                except Exception as e:
                    logging.error(f"Error processing {file_name}: {e}")

# Vectorize only the new documents
if new_documents:
    try:
        vectordb = Chroma.from_documents(
            documents=new_documents,
            embedding=embeddings,
            persist_directory="vector_db_dir"
        )
        logging.info("Documents vectorized successfully.")
    except Exception as e:
        logging.error(f"Failed to vectorize documents: {e}")
else:
    logging.info("No new documents to process.")

# Update the processed files log
with open(processed_files_log, 'w') as file:
    for file_name in processed_files:
        file.write(f"{file_name}\n")

print("Processing complete.")
