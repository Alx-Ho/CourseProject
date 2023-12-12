import os
import re

import nltk
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from openai import OpenAI
from scipy.cluster.hierarchy import fcluster

def clean_transcript(file_path, output_path):
    """
    Cleans a transcript file by removing newlines and descriptors.

    Args:
    file_path (str): The path to the transcript file to be cleaned.
    output_path (str): The path where the cleaned transcript will be saved.

    Returns:
    None
    """

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace newline characters with a space
    content = content.replace('\n', ' ')

    # Remove descriptors like [SOUND] or any all-caps text in brackets
    content = re.sub(r'\[\b[A-Z]+\b\]', '', content)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)

def get_embedding(client, sentence):
    """
    Generates an embedding for a given sentence using OpenAI's embedding model.

    Args:
    client (OpenAI): The OpenAI client instance.
    sentence (str): The sentence for which the embedding is to be generated.

    Returns:
    np.array: The generated embedding.
    """

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=sentence
    )
    return response.data[0].embedding

def sentence_tokenizer_and_embed(file_path, output_csv_path, openai_api_key=None):
    """
    Tokenizes a text file into sentences and generates embeddings for each sentence.

    Args:
    file_path (str): Path to the text file to be processed.
    output_csv_path (str): Path where the resulting CSV file will be saved.
    openai_api_key (str, optional): OpenAI API key. Defaults to None.

    Returns:
    None
    """

    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text into sentences
    sent_text = nltk.sent_tokenize(text)

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))

    # Lists to store sentences and embeddings
    data = []

    # Process each sentence
    for sentence in sent_text:
        embedding = get_embedding(client, sentence)
        data.append([sentence, *embedding])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, header=False)


def compute_linkage_matrix(embeddings_csv_path):
    """
    Computes a linkage matrix from embeddings.

    Args:
    embeddings_csv_path (str): Path to the CSV file containing embeddings.

    Returns:
    ndarray: The computed linkage matrix.
    """

    # Load the CSV file
    df = pd.read_csv(embeddings_csv_path, header=None)

    # Extract embeddings (assuming first column is the sentence)
    embeddings = df.iloc[:, 1:]

    # Compute linkage matrix
    return sch.linkage(embeddings, method='ward')


def form_clusters(linkage_matrix, relative_cutoff):
    """
    Forms clusters from a linkage matrix using a relative cutoff for distance.

    Args:
    linkage_matrix (ndarray): The linkage matrix.
    relative_cutoff (float): The relative cutoff for forming clusters.

    Returns:
    ndarray: Array of cluster labels.
    """

    # Find the minimum and maximum distances in the linkage matrix
    min_dist = np.min(linkage_matrix[:, 2])
    max_dist = np.max(linkage_matrix[:, 2])

    # Normalize the cutoff to be within the range of min_dist and max_dist
    normalized_cutoff = min_dist + relative_cutoff * (max_dist - min_dist)

    # Form flat clusters using the normalized cutoff
    return fcluster(linkage_matrix, normalized_cutoff, criterion='distance')

def summarize_cluster(sentences, cluster_indices, cluster_number, openai_api_key=None):
    """
    Summarizes a cluster of sentences into a single sentence using GPT-3.5.

    Args:
    sentences (list[str]): The list of all sentences.
    cluster_indices (list[int]): The list of cluster indices corresponding to each sentence.
    cluster_number (int): The specific cluster number to summarize.
    openai_api_key (str, optional): OpenAI API key. Defaults to None.

    Returns:
    str: The summarized sentence.
    """

    # Extract sentences belonging to the specified cluster
    cluster_sentences = [s for s, c in zip(sentences, cluster_indices) if c == cluster_number]
    cluster_text = '\n'.join(cluster_sentences)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))

    # Generate summary using GPT-3.5
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following sentences from a lecture and summarize it concisely into one sentence. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": cluster_text
            }
        ]
    )
    return response.choices[0].message.content
