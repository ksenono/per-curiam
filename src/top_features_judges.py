import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


def load_data(opinions_file):
    date_to_prompts = []
    with open(opinions_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip()) 
                prompt = data.get('text')  
                author = data.get('author') 
                date_to_prompts.append([author, prompt])


            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}\n{e}")
    return date_to_prompts


def extract_tfidf_features(prompts, ngram_range=(3, 3)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(prompts)
    return tfidf_matrix, vectorizer.get_feature_names_out()


def get_top_features_by_author(prompts, labels, top_k=1, ngram_range=(1, 1)):
    tfidf_matrix, feature_names = extract_tfidf_features(prompts, ngram_range=ngram_range)
    top_features_by_author = {}

    unique_authors = set(labels)
    for author in unique_authors:
        # Identify rows belonging to the current author
        author_indices = np.where(np.array(labels) == author)[0]
        author_tfidf_matrix = tfidf_matrix[author_indices]

        # Compute mean TF-IDF scores for this author
        mean_tfidf_scores = np.asarray(author_tfidf_matrix.mean(axis=0)).flatten()

        # Get the top-k features based on mean TF-IDF scores
        top_indices = np.argsort(mean_tfidf_scores)[::-1][:top_k]
        top_features_by_author[author] = [feature_names[idx] for idx in top_indices]

    return top_features_by_author


def main(jsonl_file):

    date_to_prompts = load_data(jsonl_file)
    
    rows = []
    labels = []
    prompts = []
    for author, prompt in date_to_prompts:
        labels.append(author)
        prompts.append(prompt)

    feature_names = get_top_features_by_author(prompts, labels, top_k=10, ngram_range=(1, 1)) 
    for author in set(labels): 
        rows.append({"Author": author, "Best features": feature_names[author]})

    feature_df = pd.DataFrame(rows)
    feature_df.sort_values(by=["Author"], ascending=[True], inplace=True)
    feature_df.reset_index(drop=True, inplace=True) 

    return feature_df


jsonl_file = '/Users/tosha/Documents/per-curiam/data/cleaned_processed.jsonl'  
feature_df = main(jsonl_file)
output_file = '/Users/tosha/Documents/per-curiam/json/features_by_year.json'
feature_df.to_json(output_file, orient='records', lines=True, indent=4)
print(tabulate(feature_df, headers="keys", tablefmt="grid"))
# plot_top_words_heatmap(feature_df)

print(f"Selected features saved to {output_file}")

