# Mock frontend to test changes

from app.cache.exact_cache import ExactCache

exact_cache = ExactCache()

query = "I have covid"
response = "Covid bad uh oh"

exact_cache.add_to_cache(query, response)

cached_response = exact_cache.check_cache(query)
if cached_response:
    print(f"Cache hit: {cached_response}")
else:
    print("Cache miss")

exact_cache.remove_from_cache(query)

exact_cache.clear_cache()
