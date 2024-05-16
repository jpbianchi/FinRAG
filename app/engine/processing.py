import os, pickle
from typing import List
from engine.loaders.file import pdf_extractor
from engine.chunk_embed import chunk_vectorize
from settings import parquet_file
from .logger import logger
from .vectorstore import VectorStore
# I allow relative imports inside the engine package
# I could have created a module but things are still changing

finrag_vectorstore = VectorStore(model_path='sentence-transformers/all-mpnet-base-v2')
    

def empty_collection():
    """ Deletes the Finrag collection if it exists """
    status = finrag_vectorstore.empty_collection()
    return status


def index_data():
    
    if not os.path.exists(parquet_file):
        logger.info(f"Parquet file {parquet_file} does not exists")
        return 'no data to index'
    
    # load the parquet file into the vectorstore
    finrag_vectorstore.index_data()
    os.remove(parquet_file)
    # delete the files so we can load several files and index them when we want
    # without having to keep track of those that have been indexed already
    # this is a simple solution for now, but we can do better
    
    return "Index creation successful"
    

def process_pdf(filepath:str) -> dict:
    
    new_content = pdf_extractor('PyPDFLoader', filepath).extract_text()
    logger.info(f"Successfully extracted text from PDF")
    
    chunk_vectorize(new_content)
    logger.info(f"Successfully vectorized PDF content")
    return new_content

def process_question(question:str) -> List[str]:
    
    ans = finrag_vectorstore.hybrid_search(query=question, limit=3, alpha=0.8)
    return ans
