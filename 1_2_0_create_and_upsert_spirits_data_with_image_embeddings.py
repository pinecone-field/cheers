"""
This Python code snippet is part of a larger script that generates image embeddings using the CLIP model and prepares the data for upsertion into a Pinecone index.

The generate_image_embedding function takes an image path as input, opens the image, converts it to RGBA if it has transparency, and then processes it using a pre-defined processor and model. The processed image features are then returned as a flattened list. If there's an error during this process, it logs the error and returns None.

After defining this function, the script logs a message indicating that it's starting to generate image embeddings and prepare the data. It then initializes two empty lists: data and skipped_rows.

The script then iterates over each row in a DataFrame spirits_data. For each row, it gets the image path from the 'imageURL' column and generates an image embedding. If the embedding is successfully generated, it creates a dictionary of metadata by dropping the 'index' and 'imageURL' columns from the row and converting the remaining columns to a dictionary. It then appends a new dictionary to the data list, with 'id' set to the 'index' of the row, 'values' set to the embedding, and 'metadata' set to the metadata dictionary. If the embedding is not successfully generated, the row is skipped.
"""
import logging
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the CSV file
csv_file_path = os.getenv("CSV_FILE")
logging.info(f"Loading the CSV file from {csv_file_path}...")
spirits_data = pd.read_csv(csv_file_path)

# Initialize the CLIP model and processor
logging.info("Initializing the CLIP model and processor...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

def generate_image_embedding(image_path):
    """
    Generate image embeddings using the CLIP model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: Image embeddings as a flattened list.
    """
    try:
        image = Image.open(image_path)
        # Convert image to RGBA if it has transparency
        if image.mode in ("P", "RGBA"):
            image = image.convert("RGBA")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

# Generate embeddings and prepare the data
logging.info("Generating image embeddings and preparing the data...")
data = []
skipped_rows = []

for idx, row in spirits_data.iterrows():
    image_path = row["imageURL"]  # Assuming 'imageURL' contains the local file path
    embedding = generate_image_embedding(image_path)
    if embedding:
        metadata = row.drop(["index", "imageURL"]).to_dict()
        data.append({"id": str(row["index"]), "values": embedding, "metadata": metadata})
    else:
        skipped_rows.append(idx)

# Convert to DataFrame
final_df = pd.DataFrame(data)

# Replace NaN values with "" in metadata
final_df["metadata"] = final_df["metadata"].apply(lambda x: {k: v if pd.notna(v) else "" for k, v in x.items()})

# Remove skipped rows from the original DataFrame
spirits_data = spirits_data.drop(skipped_rows)

# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("IMAGE_INDEX_NAME")
logging.info("Initializing Pinecone client...")
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Upsert data to Pinecone
logging.info(f"Upserting data to Pinecone index '{index_name}'...")
index.upsert_from_dataframe(final_df, batch_size=900)

logging.info("Data upserted successfully to Pinecone.")

# Save the updated DataFrame to a new CSV file
output_csv_file_path = os.getenv("OUTPUT_CSV_FILE_PATH")
logging.info(f"Saving the updated DataFrame to {output_csv_file_path}...")
spirits_data.to_csv(output_csv_file_path, index=False)

logging.info("Script completed successfully.")