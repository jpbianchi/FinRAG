
from typing import List 

from .llm import LLM
#the LLM Class uses the OPENAI_API_KEY env var as the default api_key 


def rag_it(question: str,
           search_results: List[str], 
           model: str = 'gpt-3.5-turbo-0125', 
           ) -> str:

    # TODO turn this into a class if time allows
    llm = LLM(model)

    system_message = """
    You are a financial analyst, with a deep expertise in financial reports.
    You are able to quickly understand a series of paragraphs, or quips even, extracted 
    from financial reports by a vector search system.  
    """ 
        
    searches = "\n".join([f"Search result {i}: {v}" for i,v in enumerate(search_results,1)])

    user_prompt = f"""
    Use the below context enclosed in triple back ticks to answer the question. \n
    The context is given by a vector search into a vector database of financial reports, 
    so you can assume the context is accurate.
    They search results are given in order of relevance (most relevant first). \n
    ```
    Context:
    ```
    {searches}
    ```
    Question:\n
    {question}\n
    ------------------------
    1. If the context does not provide enough information to answer the question, then
    state that you cannot answer the question with the provided context.  
    Pay great attention to making sure your answer is relevant to the question 
    (for instance, never answer a question about a topic or company that are not explicitely mentioned in the context)
    2. Do not use any external knowledge or resources to answer the question. 
    3. Answer the question directly and with as much detail as possible, within the limits of the context. 
    4. Avoid mentioning 'search results' in the answer.  
       Instead, incorporate the information from the search results into the answer.
    5. Create a clean answer, without backticks, or starting with a new line for instance.  
    ------------------------
    Answer:\n
    """.format(searches=searches, question=question)


    response = llm.chat_completion(system_message=system_message,
                                   user_message=user_prompt,
                                   temperature=0.01,  # let's not allow the model to be creative
                                   stream=False,
                                   raw_response=False)
    return response