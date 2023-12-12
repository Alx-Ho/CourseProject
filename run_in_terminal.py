# Import necessary libraries
import textwrap
import pandas as pd
from utils import clean_transcript, sentence_tokenizer_and_embed, compute_linkage_matrix, form_clusters, summarize_cluster

def main():
    # Infinite loop to continuously process transcript files until the user decides to stop
    while True:
        # Prompting the user to enter the path of the transcript text file
        input_file = input("Enter the path to the transcript text file: ")
        print("Cleaning the transcript...")
        
        # Setting the output path for the cleaned transcript
        cleaned_file = 'cleaned_transcript.txt'

        # Cleaning the transcript file
        clean_transcript(input_file, cleaned_file)

        print("Transcript cleaned.")
        # Prompting the user to enter their OpenAI API key for embedding
        openai_api_key = input("Enter your OpenAI API key to embed the transcript: ")

        # File path to save embeddings and sentences
        output_csv = 'transcript_embeddings.csv'
        print("Now tokenizing and embedding sentences...")
        
        # Tokenizing and embedding sentences in the transcript
        sentence_tokenizer_and_embed(cleaned_file, output_csv, openai_api_key=openai_api_key)

        print("Sentences tokenized and embedded. Now computing linkage matrix...")
        
        # Computing the linkage matrix for clustering
        linkage_matrix = compute_linkage_matrix(output_csv)

        # Inner loop for clustering and summarization
        while True:
            try:
                # Prompting for a relative cutoff for clustering and validating the input
                relative_cutoff = float(input("Enter a relative cutoff for clustering (0 to 1): "))
                print(f"Forming clusters using a relative cutoff of {relative_cutoff}...")
                
                # Forming clusters based on the linkage matrix and the specified cutoff
                clusters = form_clusters(linkage_matrix, relative_cutoff)

                # Displaying the number of clusters formed
                num_clusters = len(set(clusters))
                print(f'Number of clusters formed: {num_clusters}')

                # Asking user to confirm if they want to proceed with summarization
                confirm = input(f"Proceed with summarization for {num_clusters} clusters? (yes/no): ").lower()
                if confirm != 'yes':
                    continue

                print("Summarizing clusters...")
                
                # Loading sentences from the CSV file
                df = pd.read_csv(output_csv, header=None)
                sentences = df[0].tolist()

                # Summarizing each cluster and displaying the summaries
                summaries = []
                wrap_width = 80  # Width for text wrapping
                for i in range(1, num_clusters + 1):
                    summary = summarize_cluster(sentences, clusters, i, openai_api_key=openai_api_key)
                    summaries.append(summary)
                    formatted_summary = textwrap.fill(summary, width=wrap_width)
                    print(f'Cluster {i} Summary:\n{formatted_summary}\n')

                # Asking the user if they want to try another cutoff value
                next_action = input("Do you want to try another cutoff? (yes/no): ").lower()
                if next_action != 'yes':
                    break
            except ValueError:
                # Handling invalid number input for the relative cutoff
                print("Please enter a valid number for the relative cutoff.")

        # Asking the user if they want to process another transcript file
        next_action = input("Do you want to process another transcript file? (yes/no): ").lower()
        if next_action != 'yes':
            break

# Ensuring that the main function is executed only when the script is run directly
if __name__ == "__main__":
    main()
