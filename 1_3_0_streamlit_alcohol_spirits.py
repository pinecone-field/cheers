# 1_3_0_streamlit_alcohol_spirits.py

"""
This Python script uses Streamlit, OpenAI, and Pinecone to create a web application for searching spirits (alcoholic beverages).

The script begins by importing necessary libraries and loading environment variables from a .env file. It sets the OpenAI API key and initializes the OpenAI client. It also initializes the Pinecone client with the Pinecone API key and connects to a Pinecone index specified by the environment variable "SEMANTIC_INDEX_NAME".

The get_embedding function takes a text string as input and uses the OpenAI client to generate an embedding for the text using the model specified by the environment variable "OPENAI_EMBEDDING_MODEL". The embedding is returned as a list of floats.

The main function is the entry point for the Streamlit application. It sets the title of the application to "Spirit Search". The rest of the application's functionality is not shown in this excerpt.
"""
import streamlit as st
import openai
import os
from pinecone import Pinecone
from PIL import Image
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key and Pinecone API key from environment variables for security
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.Client()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("SEMANTIC_INDEX_NAME")
index = pc.Index(index_name)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")


# Function to get embeddings using OpenAI
def get_embedding(text: str) -> List[float]:
    embedding_response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    embedding = embedding_response.data[0].embedding
    return embedding


# Streamlit app
def main():
    st.title("Spirit Search")

    # Custom CSS
    st.markdown(
        """
        <style>
        .prod-name {
            height: 2em;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        </style>
    """,
        unsafe_allow_html=True,
    )

    # Search bar
    query = st.text_input("Search for products")

    if query:
        embedding = get_embedding(query)
        query_results = index.query(vector=embedding, top_k=50, include_metadata=True)

        # Initialize columns for grid display
        cols = st.columns(4)
        col_index = 0

        for match in query_results["matches"]:
            with cols[col_index]:
                # Get the filename and construct the local file path
                filename = match["metadata"]["imageURL"]

                # Load and display the image
                img = Image.open(filename)
                img = img.resize((256, 400))
                st.image(img, use_column_width=True)

                # Display product name
                product_name = match["metadata"]["Name"]
                category = match["metadata"]["Categories"]
                country = match["metadata"]["Country"]
                description = match["metadata"]["Description"]
                st.write(
                    f'<div class="prod-name">{product_name}</div>',
                    unsafe_allow_html=True,
                )
                st.write(
                    f'<div class="cat-name">{category}</div>', unsafe_allow_html=True
                )
                st.write(
                    f'<div class="country-name">{country}</div>', unsafe_allow_html=True
                )

                # Display score and ID
                score = match["score"]
                id = match["id"]
                st.write(f"Score: {score:.4f}", unsafe_allow_html=True)

            # Move to the next column
            col_index = (col_index + 1) % len(cols)


if __name__ == "__main__":
    main()
