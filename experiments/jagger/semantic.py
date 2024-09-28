from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
cache = {} # key:query, value:response, embedding

def get_input():
    return input("Enter your query: ")

def cache_response(query, response):
    query_embedding = model.encode(query)
    cache[query] = {'response': response, 'embedding': query_embedding}

def find_similar_query(query):
    query_embedding = model.encode(query)
    best_match = None
    best_similarity = 0.7

    for cached_query, data in cache.items():
        cached_embedding = data['embedding']
        similarity = util.cos_sim(query_embedding, cached_embedding)[0][0].item()
        if similarity > best_similarity:
            best_match = cached_query
            best_similarity = similarity
    
    return [best_match, best_similarity]


def main():

    while True:
        query = get_input()

        similar = find_similar_query(query)

        if similar[0]:
            print(f"Found similar query, '{similar[0]}', with similarity {similar[1]}")
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