# 1_1_1_upsert_to_pinecone_spirits_openai_text_embedding_3_large.py

"""
This Python script is used to load and preprocess a dataset of spirits
(alcoholic beverages) from a CSV file, and then upsert (insert or update) the
data to an index in Pinecone, a vector database for machine learning.

The script first loads environment variables from a .env file, including the
Pinecone API key and other configuration details. It then sets up logging.

The load_and_preprocess_data function loads a CSV file into a pandas DataFrame,
converts the 'values' column to a numpy array, renames the 'index' column to
'id', fills any missing values in the metadata columns with an empty string,
and combines the metadata fields into a dictionary in a new 'metadata' column.
It then drops the original metadata columns.

The create_or_connect_to_index function creates a new index in Pinecone or
connects to an existing one. It checks if the index already exists, and if not,
it creates the index and waits for it to be initialized. It then connects to
the index and returns it.

The main function initializes a connection to Pinecone, creates or connects to
the index, loads and preprocesses the data, views the index stats, and upserts
the data to the index. It logs a message when after the data is upserted.

The script is intended to be run as a standalone program. If it's run as a
script, it calls the main function.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the embeddings data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the embeddings data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'id', 'values', and
        'metadata' columns.
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    # Convert string to list and then to numpy array
    logger.info("Converting 'values' column to numpy array")
    df["values"] = df["values"].apply(eval).apply(np.array)

    # Rename 'index' column to 'id' and convert to string
    if "index" in df.columns:
        logger.info("Renaming 'index' column to 'id'")
        df.rename(columns={"index": "id"}, inplace=True)
        df["id"] = df["id"].astype(str)

    # Replace NaN in metadata fields with appropriate defaults
    metadata_columns = [
        "Name",
        "Country",
        "Brand",
        "Categories",
        "Tasting Notes",
        "ABV",
        "Base Ingredient",
        "Years Aged",
        "Rating",
        "Rate Count",
        "Price",
        "Volume",
        "Description",
        "status",
        "imageURL",
    ]
    logger.info("Replacing NaN values in metadata columns")

    # Fill NaN values with empty strings
    df[metadata_columns] = df[metadata_columns].fillna("")

    # Create 'metadata' column by converting metadata fields to a dictionary
    logger.info("Creating 'metadata' column")
    df["metadata"] = df[metadata_columns].apply(lambda x: x.to_dict(), axis=1)

    # Drop original metadata columns
    logger.info("Dropping original metadata columns")
    df = df.drop(metadata_columns, axis=1)

    return df


def create_or_connect_to_index(api_key, index_name, dimension, metric, spec):
    """
    Create a new index or connect to an existing index in Pinecone.

    Args:
        api_key (str): Pinecone API key.
        index_name (str): Name of the index.
        dimension (int): Dimension of the embeddings.
        metric (str): Distance metric for similarity search.
        spec (ServerlessSpec): Serverless specification for the index.

    Returns:
        Pinecone.Index: Connected Pinecone index.
    """
    pc = Pinecone(api_key=api_key)

    # Check if index already exists
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating index '{index_name}'")
        # If index does not exist, create it
        pc.create_index(
            index_name,
            dimension=dimension,
            metric=metric,
            spec=spec
        )

        # Wait for index to be initialized
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for index to be initialized...")
            time.sleep(1)
    else:
        logger.info(f"Connecting to existing index '{index_name}'")

    # Connect to the index
    index = pc.Index(index_name)
    return index


def main():
    # Initialize connection to Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("SEMANTIC_INDEX_NAME")
    cloud = os.getenv("PINECONE_CLOUD")
    region = os.getenv("PINECONE_REGION")
    metric = os.getenv("PINECONE_METRIC")
    spec = ServerlessSpec(cloud=cloud, region=region)
    dimension = os.getenv("OPENAI_MODEL_DIMENSION")

    # Create or connect to the index
    logger.info(f"Creating or connecting to index '{index_name}'")
    index = create_or_connect_to_index(
        api_key, index_name, dimension=dimension, metric=metric, spec=spec
    )

    # Load and preprocess the embeddings data
    file_path = os.getenv("UPSERT_FILE_PATH")
    logger.info(f"Loading and preprocessing data from {file_path}")
    df = load_and_preprocess_data(file_path)

    # View index stats
    logger.info("Index stats:")
    logger.info(index.describe_index_stats())

    # Upsert data to the index
    logger.info("Upserting data to the index")
    index.upsert_from_dataframe(df, batch_size=175)

    logger.info("Data upserted successfully")

    # Clean up
    # delete 1_1_0_embeddings_spirits_openai_text_embedding_3_large.csv
    os.remove(file_path)
    logger.info(f"Deleted {file_path}")
    
if __name__ == "__main__":
    main()
