import json
import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings

#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

# Retrieve selection menu for namespace and metadata
def retrieve_menu(local_menu="./data/menu.json"):
    # Check if the local file exists (useful for development)
    # if os.path.exists(local_menu):
        
    try:
        # Download from jsonsilo.com in production
        url = 'https://api.jsonsilo.com/4ced3c53-e29c-4125-b527-703df8663652'  # Replace with your actual file ID from jsonsilo.com
        headers = {
            'X-SILO-KEY': os.environ.get('JSONSILO_API_KEY'),
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception if the request fails (e.g., 404, 403)
        return response.json()
    except:
        with open(local_menu, 'r', encoding='utf-8') as file:
            return json.load(file)