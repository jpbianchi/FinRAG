---
title: FinRAG
emoji: 🐢
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

### How to use the endpoint

Please see the notebook 'app/notebooks/upload_index.ipynb' for examples of how to upload docs, index them, delete the data, erase the vector store, do a vector search or a full RAG.  

One can upload as many documents as he wants, and decide when to index them, and then continue uploading documents, and, again, index them at any time.  

The code works locally with uvicorn and here on Huggingface.  

In 'tests/test_main.py', one can find a few ideas about how to test the code.  It is of course far from being exhaustive, but I included simple unit tests and also some that test the overall capability of the code, ie answering a question with a LLM, fed by the results of a hybrid search on a Weaviate database.  