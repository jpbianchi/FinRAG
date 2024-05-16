
import os, random, logging, pickle, shutil
from dotenv import load_dotenv, find_dotenv
from typing import Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, File, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from engine.processing import process_pdf, index_data, empty_collection, vector_search
from rag.rag import rag_it

from engine.logger import logger

from settings import datadir

os.makedirs(datadir, exist_ok=True)

app = FastAPI()

environment = os.getenv("ENVIRONMENT", "dev")  # created by dockerfile

if environment == "dev":
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

try:
    load_dotenv(find_dotenv('env'))
    
except Exception as e:
    pass 


@app.get("/", response_class=HTMLResponse)
def read_root():
    logger.info("Title displayed on home page")
    return """
    <html>
        <body>
            <h1>Welcome to FinExpert, a RAG system designed by JP Bianchi!</h1>
        </body>
    </html>
    """


@app.get("/ping/")
def ping():
    """ Testing """
    logger.info("Someone is pinging the server")
    return {"answer": str(int(random.random() * 100))}


@app.delete("/erase_data/")
def erase_data():
    """ Erase all files in the data directory, but not the vector store """
    if len(os.listdir(datadir)) == 0:
        logger.info("No data to erase")
        return {"message": "No data to erase"}
    
    shutil.rmtree(datadir, ignore_errors=True)
    os.mkdir(datadir)
    logger.warning("All data has been erased")
    return {"message": "All data has been erased"}


@app.delete("/empty_collection/")
def delete_vectors():
    """ Empty the collection in the vector store """
    try:
        status = empty_collection()
        return {f"""message": "Collection{'' if status else ' NOT'} erased!"""}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/list_files/")
def list_files():
    """ List all files in the data directory """
    files = os.listdir(datadir)
    logger.info(f"Files in data directory: {files}")
    return {"files": files}


@app.post("/upload/")
# @limiter.limit("5/minute") see 'slowapi' for rate limiting
async def upload_file(file: UploadFile = File(...)):
    """  Uploads a file in data directory, for later indexing """
    try:
        filepath = os.path.join(datadir, file.filename)
        logger.info(f"Fiename detected: {file.filename}")
        if os.path.exists(filepath):
            logger.warning(f"File {file.filename} already exists: no processing done")
            return {"message": f"File {file.filename} already exists: no processing done"}    

        else:
            logger.info(f"Receiving file: {file.filename}")
            contents = await file.read()
            logger.info(f"File reception complete!")
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return {"message": f"Error during file upload:  {str(e)}"}
    
    if file.filename.endswith('.pdf'):
        
        # let's save the file in /data even if it's temp storage on HF
        with open(filepath, 'wb') as f:
            f.write(contents)
                
        try:
            logger.info(f"Starting to process {file.filename}")
            new_content = process_pdf(filepath)
            success = {"message": f"Successfully uploaded {file.filename}"}
            success.update(new_content)
            return success
        
        except Exception as e:
            return {"message": f"Failed to extract text from PDF: {str(e)}"}
    else:
        return {"message": "Only PDF files are accepted"}


@app.post("/create_index/")
async def create_index():
    """ Create an index for the uploaded files """
    
    logger.info("Creating index for uploaded files")
    try:
        msg = index_data()
        return {"message": msg}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


class Question(BaseModel):
    question: str

@app.post("/ask/")
async def hybrid_search(question: Question):
    logger.info(f"Processing question: {question.question}")
    try:
        search_results = vector_search(question.question) 
        logger.info(f"Answer: {search_results}")
        return {"answer": search_results}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    
@app.post("/ragit/")
async def ragit(question: Question):
    logger.info(f"Processing question: {question.question}")
    try:
        search_results = vector_search(question.question) 
        logger.info(f"Search results generated: {search_results}")
        
        answer = rag_it(question.question, search_results)
        
        logger.info(f"Answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
if __name__ == '__main__':
    import uvicorn
    from os import getenv
    port = int(getenv("PORT", 80))
    print(f"Starting server on port {port}")
    reload = True if environment == "dev" else False
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)



# Examples:
# curl -X POST "http://localhost:80/upload" -F "file=@test.pdf"
# curl -X DELETE "http://localhost:80/erase_data/"
# curl -X GET "http://localhost:80/list_files/" 

# hf space is at https://jpbianchi-finrag.hf.space/ 
# code given by https://jpbianchi-finrag.hf.space/docs
# Space must be public
# curl -X POST "https://jpbianchi-finrag.hf.space/upload/" -F "file=@test.pdf"

# curl -X POST http://localhost:80/ask/ -H "Content-Type: application/json" -d '{"question": "what is Amazon loss"}' 
# curl -X POST http://localhost:80/ragit/ -H "Content-Type: application/json" -d '{"question": "Does ATT have postpaid phone customers?"}'
