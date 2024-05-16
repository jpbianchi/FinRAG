import os

# from langchain.document_loaders import PyPDFLoader  # deprecated
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse  

from typing import Union, List, Dict

from abc import ABC, abstractmethod

class PDFExtractor(ABC):
    
    def __init__(self, file_or_list: Union[str, List[str]], num_workers: int = 1, verbose: bool = False):
        """ We can provide a list of files or a single file """
        if isinstance(file_or_list, str):
            self.filelist = [file_or_list]
        else:
            self.filelist = file_or_list
        self.num_workers = num_workers
        self.verbose = verbose
        super().__init__()
    
    @abstractmethod
    def extract_text(self) -> Dict[str, List[str]]:
        """ Extracts text from the PDF, no processing.
            Return a dictionary, key = filename, value = list of strings, one for each page.
        """
        pass

    @abstractmethod
    def extract_images(self):
        """Extracts images from the PDF, no processing."""
        pass

    @abstractmethod
    def extract_tables(self):
        """ Extracts tables from the PDF, no processing.
            Return in json format
        """
        pass

class _PyPDFLoader(PDFExtractor):
    
    def extract_text(self):
        output_dict = {}
        for fpath in self.filelist:
            fname = fpath.split('/')[-1]
            output_dict[fname] = [p.page_content for p in PyPDFLoader(fpath).load()]  
        return output_dict
    
    def extract_images(self):
        raise NotImplementedError("Not implemented or PyPDFLoader does not support image extraction")
        return 
    
    def extract_tables(self):
        raise NotImplementedError("Not implemented or PyPDFLoader does not support table extraction")
        return


class _LlamaParse(PDFExtractor):
    
    def extract_text(self):
        # https://github.com/run-llama/llama_parse
        if os.getenv("LLAMA_PARSE_API_KEY") is None:
            raise ValueError("LLAMA_PARSE_API_KEY is not set.")
        
        parser = LlamaParse(
            api_key = os.getenv("LLAMA_PARSE_API_KEY"),
            num_workers=self.num_workers,
            verbose=self.verbose,
            language="en",
            result_type="text"  # or "markdown"
        )
        output_dict = {}
        for fpath in self.filelist:
            # https://github.com/run-llama/llama_parse/blob/main/examples/demo_json.ipynb
            docs = parser.get_json_result(fpath)
            docs[0]['pages'][0]['text']
            output_dict[fpath] = None
        return output_dict
    
    def extract_images(self):
        raise NotImplementedError("Not implemented or LlamaParse does not support image extraction")
        return 
    
    def extract_tables(self):
        raise NotImplementedError("Not implemented or LlamaParse does not support table extraction")
        return


def pdf_extractor(extractor_type: str, *args, **kwargs) -> PDFExtractor:
    """ Factory function to return the appropriate PDF extractor instance, properly initialized """
    
    if extractor_type == 'PyPDFLoader':
        return _PyPDFLoader(*args, **kwargs)
    
    elif extractor_type == 'LlamaParse':
        return _LlamaParse(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported PDF extractor type: {extractor_type}")




