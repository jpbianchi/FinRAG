import os, logging
from typing import List, Any
import pandas as pd 
from weaviate.classes.config import Property, DataType

from .weaviate_interface_v4 import WeaviateWCS, WeaviateIndexer
from .logger import logger 

from settings import parquet_file

class VectorStore:
    def __init__(self, model_path:str = 'sentence-transformers/all-mpnet-base-v2'):
        # we can create several instances to test various models, especially if we finetune one
        
        self.finrag_properties = [  
                Property(name='filename',
                         data_type=DataType.TEXT,
                         description='Name of the file',
                         index_filterable=True,
                         index_searchable=True),
                # Property(name='keywords',
                #          data_type=DataType.TEXT_ARRAY,
                #          description='Keywords associated with the file',
                #          index_filterable=True,
                #          index_searchable=True),
                Property(name='content',
                         data_type=DataType.TEXT,
                         description='Splits of the article',
                         index_filterable=True,
                         index_searchable=True),
              ]

        self.class_name = "FinRag_all-mpnet-base-v2"

        self.class_config = {'classes': [

                            {"class": self.class_name,
                            
                            "description": "Financial reports", 
                            
                            "vectorIndexType": "hnsw", 
                            
                            # Vector index specific settings for HSNW
                            "vectorIndexConfig": {                   
                                
                                    "ef": 64,  # higher is better quality vs slower search
                                    "efConstruction": 128, # higher = better index but slower build
                                    "maxConnections": 32,  # max conn per layer - higher = more memory
                            },
                            
                            "vectorizer": "none",
                            
                            "properties": self.finrag_properties }
                            ]
        }

        self.model_path = model_path
        
        try:
            self.api_key = os.environ.get('FINRAG_WEAVIATE_API_KEY')
            self.url =  os.environ.get('FINRAG_WEAVIATE_ENDPOINT')
            self.client = WeaviateWCS(endpoint=self.url, 
                                      api_key=self.api_key, 
                                      model_name_or_path=self.model_path)
            
        except Exception as e:
            # raise Exception(f"Could not create Weaviate client: {e}")
            print(f"Could not create Weaviate client: {e}")
        
        assert self.client._client.is_live(), "Weaviate is not live"
        assert self.client._client.is_ready(), "Weaviate is not ready"
        # careful with accessing '_client' since the weaviate helper usually closes the connection every time
        
        self.indexer = None
        
        self.create_collection()
    
    @property
    def collections(self):
        
        return self.client.show_all_collections()
        
    def create_collection(self, collection_name: str='Finrag', description: str='Financial reports'):

        self.collection_name = collection_name
        if collection_name not in self.collections:
            self.client.create_collection(collection_name=collection_name, 
                                          properties=self.finrag_properties, 
                                          description=description)
            self.collection_name = collection_name
        else:
            logging.warning(f"Collection {collection_name} already exists")


    def empty_collection(self, collection_name: str='Finrag') -> bool:
        
        # not in the library yet, so I simply delete and recreate it
        if collection_name in self.collections:
            self.client.delete_collection(collection_name=collection_name)
            self.create_collection()
            return True
        else:
            logging.warning(f"Collection {collection_name} doesn't exist")
            return False


    def index_data(self, data: List[dict]= None, collection_name: str='Finrag'):
        
        if self.indexer is None:
            self.indexer = WeaviateIndexer(self.client)
        
        if data is None:
            # use the parquet file, otherwise use the data passed
            data = pd.read_parquet(parquet_file).to_dict('records')
            # the parquet file was created/incremented when a new article was uploaded
            # it is a dataframe with columns: file, content, content_embedding
            # and reflects exactly the data that we want to index at all times
        self.status = self.indexer.batch_index_data(data, collection_name, 256)
        
        self.num_errors, self.error_messages, self.doc_ids = self.status
        
        # in this case with few articles, we don't tolerate errors
        # batch_index_data already tests errors against a threshold
        # assert self.num_errors == 0, f"Errors: {self.num_errors}"
        
        
    def keyword_search(self, 
                       query: str, 
                       limit: int=5, 
                       return_properties: List[str]=['filename', 'content'],
                       alpha=None  # dummy parameter to match the hybrid_search signature
                       ) -> List[str]:
        response = self.client.keyword_search(
                                request=query,
                                collection_name=self.collection_name,
                                query_properties=['content'], 
                                limit=limit,
                                filter=None,  
                                return_properties=return_properties,
                                return_raw=False)
        
        return [res['content'] for res in response]
    
    
    def vector_search(self, 
                      query: str, 
                      limit: int=5, 
                      return_properties: List[str]=['filename', 'content'],
                      alpha=None  # dummy parameter to match the hybrid_search signature
                      ) -> List[str]:
        
        response = self.client.vector_search(
                                request=query,
                                collection_name=self.collection_name,
                                limit=limit,
                                filter=None,  
                                return_properties=return_properties,
                                return_raw=False)
        
        return [res['content'] for res in response]
    
    
    def hybrid_search(self, 
                      query: str, 
                      limit: int=5, 
                      alpha=0.5,  # higher = more vector search
                      return_properties: List[str]=['filename', 'content']
                      ) -> List[str]:

        response = self.client.hybrid_search(
                                request=query,
                                collection_name=self.collection_name,
                                query_properties=['content'],
                                alpha=alpha,  
                                limit=limit,
                                filter=None,  
                                return_properties=return_properties,
                                return_raw=False)
        
        return [res['content'] for res in response]