import textwrap
import pandas as pd

from utils import clean_transcript, sentence_tokenizer_and_embed, compute_linkage_matrix, form_clusters, summarize_cluster

def main():
    while True:
        input_file = input("Enter the path to the transcript text file: ")
        print("Cleaning the transcript...")
        cleaned_file = 'results/cleaned_transcript.txt'  # Output path for the cleaned transcript

        clean_transcript(input_file, cleaned_file)

        print("Transcript cleaned.")
        openai_api_key = input("Enter your OpenAI API key to embed the transcript: ")

        output_csv = 'results/transcript_embeddings.csv'  # File path to save embeddings and sentences
        print("Now tokenizing and embedding sentences...")
        sentence_tokenizer_and_embed(cleaned_file, output_csv, openai_api_key=openai_api_key)

        print("Sentences tokenized and embedded. Now computing linkage matrix...")
        linkage_matrix = compute_linkage_matrix(output_csv)

        while True:
            try:
                relative_cutoff = float(input("Enter a relative cutoff for clustering (0 to 1): "))
                print(f"Forming clusters using a relative cutoff of {relative_cutoff}...")
                clusters = form_clusters(linkage_matrix, relative_cutoff)

                # Display the number of clusters formed
                num_clusters = len(set(clusters))
                print(f'Number of clusters formed: {num_clusters}')

                confirm = input(f"Proceed with summarization for {num_clusters} clusters? (yes/no): ").lower()
                if confirm != 'yes':
                    continue

                print("Summarizing clusters...")
                # Load sentences
                df = pd.read_csv(output_csv, header=None)
                sentences = df[0].tolist()

                # Summarize each cluster
                summaries = []
                wrap_width = 80
                for i in range(1, num_clusters + 1):
                    summary = summarize_cluster(sentences, clusters, i, openai_api_key=openai_api_key)
                    summaries.append(summary)
                    formatted_summary = textwrap.fill(summary, width=wrap_width)
                    print(f'Cluster {i} Summary:\n{formatted_summary}\n')

                next_action = input("Do you want to try another cutoff? (yes/no): ").lower()
                if next_action != 'yes':
                    break
            except ValueError:
                print("Please enter a valid number for the relative cutoff.")

        next_action = input("Do you want to process another transcript file? (yes/no): ").lower()
        if next_action != 'yes':
            break

if __name__ == "__main__":
    main()
