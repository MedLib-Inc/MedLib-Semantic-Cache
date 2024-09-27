import numpy as np
import pandas as pd
import chromadb
import json
import time
import faiss
from sentence_transformers import SentenceTransformer

from huggingface_hub import login
from datasets import load_dataset
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class startup:
    MAX_ROWS = 15000
    DOCUMENT="Answer"
    TOPIC = "qtype"

    #Enter your own huggingface token
    login(token="")

    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')

    data = data.to_pandas()
    data["id"]=data.index
    data.head(10)

    subset_data = data.head(MAX_ROWS)

    chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")

    collection_name = "news_collection"
    if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.create_collection(name=collection_name)

    # Function to split data into smaller batches
    def batchify(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    # Define your batch size (adjust this based on your environment's limits)
    BATCH_SIZE = 100  # Example batch size, can be adjusted

    # Get the total number of rows
    total_rows = min(len(subset_data), MAX_ROWS)

    # Prepare the documents, metadata, and IDs for batching
    documents = subset_data[DOCUMENT].tolist()
    metadatas = [{"qtype" : topic} for topic in subset_data[TOPIC].tolist()]
    ids = [f"id{x}" for x in range(total_rows)]

    # Iterate over batches
    print("Started adding batches to collection...\n")
    for batch_documents, batch_metadatas, batch_ids in zip(
        batchify(documents, BATCH_SIZE),
        batchify(metadatas, BATCH_SIZE),
        batchify(ids, BATCH_SIZE)
    ):

        # Add each batch to the collection
        collection.add(
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    print("Finished adding batches to collection...\n")

    def query_database(query_text, n_results=10):
        results = startup.collection.query(query_texts=query_text, n_results=n_results )
        return results

    def init_cache():
        index = faiss.IndexFlatL2(768)
        if index.is_trained:
            print('Index trained')
        else:
            print('Index not trained')
        encoder = SentenceTransformer('all-mpnet-base-v2')
        return index, encoder

    def retrieve_cache(json_file):
        try:
            with open(json_file, 'r') as file:
                cache = json.load(file)
        except FileNotFoundError:
            cache = {'questions': [], 'embeddings': [], 'answers': [], 'response_text': []}
        return cache

    def store_cache(json_file, cache):
        with open(json_file, 'w') as file:
            json.dump(cache, file)

    def query_database(query_text, n_results=10):
        results = startup.collection.query(query_texts=query_text, n_results=n_results )
        return results

class semantic_cache:
  def __init__(self, json_file="cache_file.json", thresold=0.35, max_response=100, eviction_policy=None):
    """Initializes the semantic cache.

    Args:
    json_file (str): The name of the JSON file where the cache is stored.
    thresold (float): The threshold for the Euclidean distance to determine if a question is similar.
    max_response (int): The maximum number of responses the cache can store.
    eviction_policy (str): The policy for evicting items from the cache.
    This can be any policy, but 'FIFO' (First In First Out) has been implemented for now.
    If None, no eviction policy will be applied.
    """

    # Initialize Faiss index with Euclidean distance
    self.index, self.encoder = startup.init_cache()

    # Set Euclidean distance threshold
    # a distance of 0 means identicals sentences
    # We only return from cache sentences under this thresold
    self.euclidean_threshold = thresold

    self.json_file = json_file
    self.cache = startup.retrieve_cache(self.json_file)
    self.max_response = max_response
    self.eviction_policy = eviction_policy

  def evict(self):

    """Evicts an item from the cache based on the eviction policy."""
    if self.eviction_policy and len(self.cache["questions"]) > self.max_size:
        for _ in range((len(self.cache["questions"]) - self.max_response)):
            if self.eviction_policy == 'FIFO':
                self.cache["questions"].pop(0)
                self.cache["embeddings"].pop(0)
                self.cache["answers"].pop(0)
                self.cache["response_text"].pop(0)

  def ask(self, question: str) -> str:
      # Method to retrieve an answer from the cache or generate a new one
      start_time = time.time()
      try:
          #First we obtain the embeddings corresponding to the user question
          embedding = self.encoder.encode([question])

          # Search for the nearest neighbor in the index
          self.index.nprobe = 8
          D, I = self.index.search(embedding, 1)

          if D[0] >= 0:
              if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                  row_id = int(I[0][0])

                  #print('Answer recovered from Cache. ')
                  #print(f'{D[0][0]:.3f} smaller than {self.euclidean_threshold}')
                  #print(f'Found cache in row: {row_id} with score {D[0][0]:.3f}')
                  #print(f'response_text: ' + self.cache['response_text'][row_id])

                  #end_time = time.time()
                  #elapsed_time = end_time - start_time
                  #print(f"Time taken: {elapsed_time:.3f} seconds")
                  return "Answer recovered from Cache: " + self.cache['response_text'][row_id]

          # Handle the case when there are not enough results
          # or Euclidean distance is not met, asking to chromaDB.
          answer = startup.query_database([question], 1)
          response_text = answer['documents'][0][0]

          self.cache['questions'].append(question)
          self.cache['embeddings'].append(embedding[0].tolist())
          self.cache['answers'].append(answer)
          self.cache['response_text'].append(response_text)

          #print('Answer recovered from ChromaDB. ')
          #print(f'response_text: {response_text}')

          self.index.add(embedding)

          self.evict()

          startup.store_cache(self.json_file, self.cache)

          #end_time = time.time()
          #elapsed_time = end_time - start_time
          #print(f"Time taken: {elapsed_time:.3f} seconds")

          return "Answer recovered from ChromaDB: " + response_text
      except Exception as e:
          raise RuntimeError(f"Error during 'ask' method: {e}")

cache = semantic_cache('4cache.json')

@app.get("/ask/{question}")
def ask_cache(question: str):
    result = cache.ask(question)
    return result