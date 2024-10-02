import json
import time
import pandas as pd
import chromadb

from huggingface_hub import login
from datasets import load_dataset
from fastapi import FastAPI

app = FastAPI()

class Startup:
    MAX_ROWS = 100
    DOCUMENT = "Answer"
    TOPIC = "qtype"
    #login token from huggingface
    login(token="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")
    #load dataset from huggingface
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')

    data = data.to_pandas()
    data["id"]=data.index
    
    subset_data = data.head(MAX_ROWS)
    print(subset_data.head())
    chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")

    collection_name = "exact_cache"
    if len(chroma_client.list_collections()) > 0 or collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)
    
    collection = chroma_client.create_collection(name=collection_name)

    total_rows = min(len(subset_data),MAX_ROWS)

    documents = subset_data[DOCUMENT].tolist()
    metadatas = [{"qtype" : topic} for topic in subset_data[TOPIC].tolist()]
    ids = [f"id{x}" for x in range(total_rows)]

    BATCH_SIZE = 100

    def batchify(data, batch_size):
        for i in range(0,len(data),batch_size):
            yield data[i:i+batch_size]

    for batch_documents, batch_metadatas, batch_ids in zip(
        batchify(documents, BATCH_SIZE),
        batchify(metadatas, BATCH_SIZE),
        batchify(ids, BATCH_SIZE),
    ):
        collection.add(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    def query_database(query_text):
        if query_text[0] in Startup.documents:
            index = Startup.documents.index(query_text[0])  
            return [Startup.documents[index]]
        return []    

class exact_cache:
    def __init__(self, json_file = "cache_file.json", max_responses = 100):
        """"Initializes the Exact Cache
        
        Args:
        json_file (str): Name of JSON file where the cache is stored
        max_responses (int): Maximum number of responses the cache can store
        """

        self.json_file = json_file
        self.cache = self.retrieve_cache()
        self.max_responses = max_responses

    def retrieve_cache(self):
        try:
            with open(self.json_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return {'questions': [], 'answers': []}
        
    def store_cache(self):
        with open(self.json_file, 'w') as file:
            json.dump(self.cache, file)

    def add_to_cache(self,question: str, answer: str):
        if question not in self.cache['questions']:
            self.cache['questions'].append(question)
            self.cache['answers'].append(answer)

            if(len(self.cache['questions']) > self.max_responses):
                self.evict()

            self.store_cache()
    
    def evict(self):
        if len(self.cache['questions']) > 0:
            self.cache['questions'].pop(0)
            self.cache['answers'].pop(0)


    def get_answer(self, question: str):
        if question in self.cache['questions']:
            index = self.cache['questions'].index(question)
            return self.cache['answers'][index]
        else:
            return None
    
cache = exact_cache()

@app.get("/ask/{question}")
def ask_cache(question:str):
    result = cache.get_answer(question)
    if result is not None:
        return {"answer":result}

    #Search collection / generate response
    search_result = Startup.query_database([question])
    if search_result:
        answer = search_result[0]

        cache.add_to_cache(question, answer)
        return {"answer": answer}
    print("test")
    cache.add_to_cache(question, "this is a cached response")
    return {"answer": "generate new response"}