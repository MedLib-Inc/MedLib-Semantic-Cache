import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

# initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# initialize dictionary for caching
cache = {} # key:query, value:response, embedding

# dimension of MiniLM embeddings
index = faiss.IndexFlatL2(384)

def get_input():
    return input("Enter your query: ")

def cache_response(query, response):
    query_embedding = model.encode(query)

    # add embedding to cache
    cache[query] = {'response': response, 'embedding': query_embedding}

    # add embedding to Faiss index
    index.add(np.array([query_embedding]))

def find_similar_query(query):
    query_embedding = model.encode(query)

    # search for closes neighbor
    D, I = index.search(np.array([query_embedding]), 1)

    distance_threshold = 0.35

    if D[0][0] < distance_threshold:
        similar = list(cache.keys())[I[0][0]]
        return [similar, D[0][0]]
    else:
        return[None, None]


def main():

    while True:
        query = get_input()

        similar = find_similar_query(query)

        if similar[0]:
            print(f"Found similar query, '{similar[0]}', with Euclidean distance {similar[1]}")
            print(f"Reponse: {cache[similar[0]]['response']}")
        else:
            response = f"simulated response for '{query}'"
            cache_response(query, response)
            print(f"Response: {response}")

        if query.lower() == "exit":
            print("exiting...")
            break


if __name__ == "__main__":
    main()