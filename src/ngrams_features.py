import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Preprocess the dataset 
def load_data(opinions_file):
    prompts, completions = [], []
    with open(opinions_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                prompts.append(data['prompt'])       
                completions.append(data['completion'])  
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}\n{e}")
    return prompts, completions

# Feature extraction using TF-IDF
def extract_tfidf_features(texts, ngram_range=(3, 3)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer.get_feature_names_out()

# Calculate Information Gain
def select_features_using_ig(tfidf_matrix, labels, top_k=500):
    tfidf_dense = tfidf_matrix.toarray()
    # Calculate mutual information (IG)
    ig_scores = mutual_info_classif(tfidf_dense, labels, discrete_features=False)
    # Get indices of top features
    top_indices = np.argsort(ig_scores)[-top_k:]
    return top_indices, ig_scores

# Main workflow
def main(jsonl_file, top_k=500):
    # Load data
    prompts, completions = load_data(jsonl_file)

    # Encode labels
    unique_labels = list(set(completions))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_mapping[completion] for completion in completions]

    # Extract TF-IDF features
    tfidf_matrix, feature_names = extract_tfidf_features(prompts, ngram_range=(3, 3))

    # Select features using IG
    top_indices, ig_scores = select_features_using_ig(tfidf_matrix, labels, top_k=top_k)
    selected_features = [feature_names[idx] for idx in top_indices]

    # Create a DataFrame for the selected features and their IG scores
    feature_df = pd.DataFrame({
        'Feature': selected_features,
        'Information Gain': ig_scores[top_indices]
    }).sort_values(by='Information Gain', ascending=False)

    return feature_df, tfidf_matrix, labels, top_indices

# Run the script
jsonl_file = '/Users/tosha/Documents/per-curiam/data/opinions.jsonl'  
feature_df, tfidf_matrix, labels, top_indices = main(jsonl_file, top_k=500)

# Display selected features
# print(feature_df)

# Save the result to a JSON file
# output_file = '/Users/tosha/Documents/per-curiam/json/features_unigrams.json'  # Specify the output file name
output_file = '/Users/tosha/Documents/per-curiam/json/features_bigrams.json'
feature_df.to_json(output_file, orient='records', lines=True, indent=4)

print(f"Selected features saved to {output_file}")

