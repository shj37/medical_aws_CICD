from langchain_community.embeddings import HuggingFaceEmbeddings
import json

#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings

# retrieve selection menu for namespace and metadata
def retrieve_menu():
    with open("menu.json", 'r', encoding='utf-8') as file:
        menu_content = json.load(file)
    return menu_content