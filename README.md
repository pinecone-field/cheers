# Cheers Project Summary
The Cheers project comprises a series of Python scripts developed to construct a semantic and image search index for an extensive dataset of spirits (alcoholic beverages). Utilizing advanced models from OpenAI and CLIP, the project processes and indexes detailed information about various spirits. This enables efficient searches based on multiple criteria, including name, category, country of origin, and visual appearance, thereby facilitating comprehensive data accessibility and retrieval for researchers and industry professionals.

## Owner
Kevin M. Butler, Customer Success Engineer

## Key Components

- **CSV Dataset**: [`1_0_0_spirits_data_with_imageURLs.csv`](1_0_0_spirits_data_with_imageURLs.csv)
  - Contains detailed information about various spirits, including name, country of origin, brand, categories, ABV, rating, price, and an image URL.

- **Text Embeddings Script**: [`1_1_0_create_embeddings_spirits_openai_text_embedding_3_large.py`](1_1_0_create_embeddings_spirits_openai_text_embedding_3_large.py)
  - Uses OpenAI's text embedding model to generate embeddings for the spirits dataset.
  - Loads the data, preprocesses it, generates embeddings, and writes the results back to a CSV file.

- **Pinecone Upsert Script**: [`1_1_1_upsert_to_pinecone_spirits_openai_text_embedding_3_large.py`](1_1_1_upsert_to_pinecone_spirits_openai_text_embedding_3_large.py)
  - Loads and preprocesses the spirits dataset, then upserts the data to a Pinecone index.
  - Includes functions for data loading, preprocessing, and upsertion.

- **Image Embeddings Script**: [`1_2_0_create_and_upsert_spirits_data_with_image_embeddings.py`](1_2_0_create_and_upsert_spirits_data_with_image_embeddings.py)
  - Generates image embeddings using the CLIP model and prepares the data for upsertion into Pinecone.
  - Processes each image, generates an embedding, and appends it to a list.

## Environment Setup
```
conda create -n cheers_venv python=3.10
conda activate cheers_venv
pip install -r requirements.txt
```

An alternative to installing dependencies is to use Python virtual environments. Virtual environments allow you to manage project-specific dependencies, which can help prevent conflicts between different projects.

Here's how you can create a virtual environment and install the dependencies using the `requirements.txt` file:

1. **Create a virtual environment**: Navigate to your project directory in the terminal and run the following command to create a virtual environment named 'venv':

    ```bash
    python3 -m venv cheers_venv
    ```

    This command creates a new directory named 'cheers_venv' in your project directory, which contains the Python executable and other necessary files.

2. **Activate the virtual environment**: Before you can use the virtual environment, you need to activate it. The command to do this depends on your operating system:

    - On macOS and Linux:

        ```bash
        source cheers_venv/bin/activate
        ```

    When the virtual environment is activated, your terminal prompt changes to show the name of the virtual environment. This indicates that when you run Python, it uses the version in the virtual environment and installs packages to the virtual environment.

3. **Install the dependencies**: With the virtual environment activated, you can install the dependencies using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    This command installs all the packages listed in the `requirements.txt` file to the virtual environment.

Remember to activate the virtual environment every time you work on your project. When you're done, you can deactivate it by running the `deactivate` command.

## Environment Variables
Copy the .env_sampe to .env and update the settings.

## Images files
Download the [images.zip](https://drive.google.com/file/d/1bDE5SiVFCJOowuQ-eGDsoF2OL1Iwa4Ea/view?usp=drive_link) file and place the /images directory in the root of your project. 

## Note  
All scripts are designed to be run in sequence.

