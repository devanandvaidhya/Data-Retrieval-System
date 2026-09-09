# import os
# from huggingface_hub import InferenceClient

# client = InferenceClient(
#     api_key=os.environ["HF_TOKEN"]
# )

# embedding = client.feature_extraction(
#     "What is artificial intelligence?",
#     model="BAAI/bge-small-en-v1.5"
# )

# print(embedding)


import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from groq import Groq
load_dotenv()

token = os.getenv("GROQ_API_KEY")

# client = InferenceClient(
#     api_key=token
# )

# text = "What is artificial intelligence?"

# embedding = client.feature_extraction(
#     text,
#     model="BAAI/bge-small-en-v1.5"
# )

# print("Embedding generated successfully!")
# print("Type:", type(embedding))
# print("Embedding:", embedding)



# client = Groq(api_key=token)

# response = client.models.list()

# print(response)

import os

key = os.getenv("GROQ_API_KEY")
print("Key exists:", bool(key))
print("Starts with gsk_:", key.startswith("gsk_") if key else False)
print("Length:", len(key) if key else 0)