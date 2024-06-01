# 1_1_0_create_embeddings_spirits_openai_text_embedding_3_large.py

"""
This Python script is used to generate embeddings for a dataset of spirits (alcoholic beverages) using OpenAI's text embedding model.

The script first loads environment variables from a .env file, including the OpenAI API key and the embedding model to use. It then sets up logging.

The load_and_preprocess_data function loads a CSV file into a pandas DataFrame, selects specific columns, fills any missing values with an empty string, and combines the values of all columns into a single string for each row.

The get_embeddings_batch function sends a batch of texts to the OpenAI API to generate embeddings. It includes retry logic in case of failure.

The generate_embeddings function iterates over the DataFrame in batches, generates embeddings for each batch, and adds them to the DataFrame. If the process is interrupted, it logs a message and raises an exception.

The main function gets the input and output file paths from environment variables, loads and preprocesses the data, generates embeddings, and writes the DataFrame with embeddings to a CSV file. It handles keyboard interrupts and logs a message when processing is completed.

The script is intended to be run as a standalone program. If it's run as a script, it calls the main function.
"""
import pandas as pd
from openai import OpenAI
import openai
import numpy as np
import time
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
client = OpenAI()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_and_preprocess_data(input_datapath):
    """
    Load and preprocess the dataset.

    Args:
        input_datapath (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = pd.read_csv(input_datapath, index_col=0)
    df = df[
        [
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
    ]
    df = df.fillna("")
    df["combined"] = (
        "Name: "
        + df.Name.str.strip()
        + "; Country: "
        + df.Country.str.strip()
        + "; Brand: "
        + df.Brand.str.strip()
        + "; Categories: "
        + df.Categories.str.strip()
        + "; Tasting Notes: "
        + df["Tasting Notes"].str.strip()
        + "; ABV: "
        + df.ABV.str.strip()
        + "; Base Ingredient: "
        + df["Base Ingredient"].str.strip()
        + "; Years Aged: "
        + df["Years Aged"].astype(str).str.strip()
        + "; Rating: "
        + df.Rating.astype(str).str.strip()
        + "; Rate Count: "
        + df["Rate Count"].astype(str).str.strip()
        + "; Price: "
        + df.Price.str.strip()
        + "; Volume: "
        + df.Volume.str.strip()
        + "; Description: "
        + df.Description.str.strip()
    )
    return df


def get_embeddings_batch(texts, retries=5):
    """
    Get embeddings for a batch of texts from OpenAI with retry logic.

    Args:
        texts (list): The input list of texts to generate embeddings for.
        retries (int): The number of times to retry in case of failure.

    Returns:
        list: The list of embedding vectors.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(input=texts, model=embedding_model)
            return [data.embedding for data in response.data]
        except (openai.Timeout, openai.APIError, openai.APIConnectionError) as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                logging.error(f"All {retries} attempts failed.")
                return [np.nan] * len(texts)


def generate_embeddings(df, batch_size=2000):
    """
    Generate embeddings for the combined text in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        batch_size (int): The batch size for API requests.

    Returns:
        pd.DataFrame: The DataFrame with embeddings added.
    """
    embeddings = []

    try:
        for start in tqdm(range(0, len(df), batch_size)):
            end = start + batch_size
            batchText = df["combined"][start:end].tolist()

            if batchText:
                batch_embeddings = get_embeddings_batch(batchText)
            else:
                batch_embeddings = [np.nan] * batch_size

            embeddings.extend(batch_embeddings)

    except KeyboardInterrupt:
        logging.info("Processing interrupted.")
        raise

    # Truncate to match the DataFrame length
    df["values"] = embeddings[: len(df)]

    # drop 'combined' column
    df = df.drop(columns=["combined"])
    return df


def main():
    input_datapath = os.getenv("CSV_FILE")
    ouput_datapath = os.getenv("OUTPUT_CSV_FILE_PATH_SEMANTIC")
    df = load_and_preprocess_data(input_datapath)

    try:
        df = generate_embeddings(df)
        df.to_csv(ouput_datapath, index=True)
    except KeyboardInterrupt:
        logging.info("Main process interrupted. Exiting...")
    finally:
        logging.info("Processing completed.")


if __name__ == "__main__":
    main()
