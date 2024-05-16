

import os
import pandas as pd
import torch

from settings import parquet_file

import tiktoken  # tokenizer library for use with OpenAI LLMs 
from llama_index.legacy.text_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create tensors on GPU if available
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def chunk_vectorize(doc_content: dict = None, 
                    chunk_size: int = 256,    # limit for 'all-mpnet-base-v2'
                    chunk_overlap: int = 20,  # some overlap to link the chunks
                    encoder: str = 'gpt-3.5-turbo-0613',
                    model_name: str = 'sentence-transformers/all-mpnet-base-v2'):  # can try all-MiniLM-L6-v2
    # see tests in chunking_indexing.ipynb for more details

    encoding = tiktoken.encoding_for_model(encoder)

    splitter = SentenceSplitter(chunk_size=chunk_size, 
                                tokenizer=encoding.encode, 
                                chunk_overlap=chunk_overlap)

    # let's create the splits for every document
    contents_splits = {}
    for fname, content in doc_content.items():
        splits = [splitter.split_text(page) for page in content]
        contents_splits[fname] = [split for sublist in splits for split in sublist]
        
    model = SentenceTransformer(model_name)

    content_emb = {}
    for fname, splits in contents_splits.items():
        content_emb[fname] = [(split, model.encode(split)) for split in splits]

    # save fname since it carries information, and could be used as a property in Weaviate
    text_vector_tuples = [(fname, split, emb.tolist()) for fname, splits_emb in content_emb.items() for split, emb in splits_emb]

    new_df = pd.DataFrame(
        text_vector_tuples, 
        columns=['file', 'content', 'content_embedding']
    )
    
    # load the existing parquet file if it exists and update it 
    if os.path.exists(parquet_file):
        new_df = pd.concat([pd.read_parquet(parquet_file), new_df])

    # no optimization here (zipping etc) since the data is small
    new_df.to_parquet(parquet_file, index=False)        
    
    return

# TODO 
# import unittest
# from unitesting_utils import load_impact_theory_data

# class TestSplitContents(unittest.TestCase):
#     '''
#     Unit test to ensure proper functionality of split_contents function
#     '''
    
#     def test_split_contents(self):
#         import tiktoken
#         from llama_index.text_splitter import SentenceSplitter
        
#         data = load_impact_theory_data()
                
#         subset = data[:3]
#         chunk_size = 256
#         chunk_overlap = 0
#         encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')
#         gpt35_txt_splitter = SentenceSplitter(chunk_size=chunk_size, tokenizer=encoding.encode, chunk_overlap=chunk_overlap)
#         results = split_contents(subset, gpt35_txt_splitter)
#         self.assertEqual(len(results), 3)
#         self.assertEqual(len(results[0]), 83)
#         self.assertEqual(len(results[1]), 178)
#         self.assertEqual(len(results[2]), 144)
#         self.assertTrue(isinstance(results, list))
#         self.assertTrue(isinstance(results[0], list))
#         self.assertTrue(isinstance(results[0][0], str))
# unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestSplitContents))
