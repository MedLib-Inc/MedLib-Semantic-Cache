import logging
import requests
import os
from .persistence import Chroma
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables.")

class SemanticCache:
    def __init__(self, threshold=0.15):
        """
        Initializes the semantic cache with ChromaDB and a similarity threshold
        """
        self.persistence = Chroma(threshold=threshold) # Pass threshold to ChromaDB client

        # Set up logging globally in the entry point
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("SemanticCache initialized.")
        
    def add_to_cache(self, query, response):
        """
        Add query and response to the persistent cache (ChromaDB).
        """
        # Store in ChromaDB
        self.persistence.add_to_db(query, response)

    def check_cache(self, query):
        """
        Check if a semantically similar response is in the cache.
        """
        # Query ChromaDB for similar response
        results = self.persistence.query_db(query)

        # Ensure results contain valid data
        if not results or 'distances' not in results or not results['distances']:
            logging.warning(f"No valid results found for query: '{query}'")
            return None

        # Extract the top distance safely
        top_distance = results['distances'][0][0]
        logging.info(f"Top distance: {top_distance} (Threshold: {self.persistence.threshold})")

        # Apply threshold comparison (1 - cosine similarity)
        if top_distance <= self.persistence.threshold:
            logging.info(f"Cache hit for query: '{query}'")
            return results['documents'][0]
        else:
            logging.info(f"Cache miss: Distance {top_distance} exceeds threshold.")
        return None


    def ask(self, query):
        """
        Main method to handle a query: check cache or return a new response.
        Returns cached response if available, otherwise queries LLM.
        """
        cached_response = self.check_cache(query)

        if cached_response:
            # Return cached response if found
            return cached_response

        # If not found in the cache, simulate a calling the LLM
        response = self.query_external(query)

        # Add the new response to the cache
        self.add_to_cache(query, response)

        return response

    def query_external(self, query):
        """
        Call the LLM for a new response if one is not found in the cache.
        Using Hugging Face API for now, google/flan-t5 model
        """
        logging.info(f"Calling LLM for query: '{query}'")

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}"
        }
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

        payload = {"inputs": query}
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for failed requests
            result = response.json()
            if result and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text']
                logging.info(f"LLM response: '{generated_text}' for query: '{query}'")
                return generated_text
            else:
                logging.warning(f"No valid response from LLM for query: '{query}'")
                return f"No valid response generated for: {query}"
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            return f"Error generating response for: {query}"

    def update_size(self, size):
        """
        Update size of cache with the inputted size, calls method in persistence
        to reinitialize the client with the updated size. Unable to change while
        client is active without unintended consequences.
        """
        self.persistence.change_size(size)
