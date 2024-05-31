# Cheers Project Summary
The Cheers project is a collection of Python scripts designed to create a semantic and image search index for a dataset of spirits (alcoholic beverages).

## Owner
Kevin M. Butler, Customer Success Engineer

## Key Components

- **CSV Dataset**: `1_0_0_spirits_data_with_imageURLs.csv`
  - Contains detailed information about various spirits, including name, country of origin, brand, categories, ABV, rating, price, and an image URL.

- **Text Embeddings Script**: `1_1_0_create_embeddings_spirits_openai_text_embedding_3_large.py`
  - Uses OpenAI's text embedding model to generate embeddings for the spirits dataset.
  - Loads the data, preprocesses it, generates embeddings, and writes the results back to a CSV file.

- **Pinecone Upsert Script**: `1_1_1_upsert_to_pinecone_spirits_openai_text_embedding_3_large.py`
  - Loads and preprocesses the spirits dataset, then upserts the data to a Pinecone index.
  - Includes functions for data loading, preprocessing, and upsertion.

- **Image Embeddings Script**: `1_2_0_create_and_upsert_spirits_data_with_image_embeddings.py`
  - Generates image embeddings using the CLIP model and prepares the data for upsertion into Pinecone.
  - Processes each image, generates an embedding, and appends it to a list.

## Environment Setup
```
conda create -n cheers_venv python=3.10
conda activate cheers
conda activate cheers
conda activate cheers_venv
pip install -r requirements.txt
```
Copy the .env_sampe to .env and update the settings.

This code was tested with Python 3.10

## Note  
All scripts are designed to be run in sequence.
