# Exact Cache main file

import redis
import json

class ExactCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.cache = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def add_to_cache(self, query, response):
        try:
            self.cache.set(query, json.dumps(response))
        except Exception as e:
            print(f"Error storing data in cache: {e}")

    def check_cache(self, query):
        try:
            # Check if the query exists in the cache
            cached_response = self.cache.get(query)
            if cached_response:
                return json.loads(cached_response)
        except Exception as e:
            print(f"Error accessing cache: {e}")
        
        return None

    def remove_from_cache(self, query):
        try:
            self.cache.delete(query)
        except Exception as e:
            print(f"Error removing data from cache: {e}")

    def clear_cache(self):
        try:
            self.cache.flushdb()
        except Exception as e:
            print(f"Error clearing cache: {e}")
