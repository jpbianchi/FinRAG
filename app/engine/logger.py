import os, logging

environment = os.getenv("ENVIRONMENT", "dev")  # TODO put the logger creation in its own file
if environment == "dev":
    logger = logging.getLogger("uvicorn")
else:
    logger = lambda x: _
    # we should log also in production  TODO 
    # check how it works on HuggingFace, if possible
    # because we don't have access to the container's file system