import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

# Load data from the JSONL file
def load_data(jsonl_file):
    prompts, completions = [], []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
            completions.append(data['completion'])
    return prompts, completions

# Split the dataset
def split_dataset(prompts, completions, test_size=0.2, val_size=0.1, random_state=42):
    # Split into train+val and test
    prompts_train_val, prompts_test, completions_train_val, completions_test = train_test_split(
        prompts, completions, test_size=test_size, random_state=random_state, stratify=completions
    )
    # Split train+val into train and val
    val_relative_size = val_size / (1 - test_size)  # Adjust validation size relative to train+val
    prompts_train, prompts_val, completions_train, completions_val = train_test_split(
        prompts_train_val, completions_train_val, test_size=val_relative_size, random_state=random_state, stratify=completions_train_val
    )
    return prompts_train, prompts_val, prompts_test, completions_train, completions_val, completions_test

# Perform DF + Information Gain feature selection
def select_features_df_ig(prompts_train, completions_train, ngram_range=(2, 2), top_k=5000):
    # Compute Document Frequency (DF) using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range, binary=True, min_df=5, max_df=0.9)
    X_train = vectorizer.fit_transform(prompts_train)
    svd = TruncatedSVD(n_components=500)
    X_train_reduced = svd.fit_transform(X_train)
    ig_scores = mutual_info_classif(X_train_reduced, completions_train, discrete_features=True)
        # Compute Information Gain (mutual information)
    
    # Select top-k features based on IG scores
    top_indices = np.argsort(ig_scores)[-top_k:]
    X_train_selected = X_train[:, top_indices]

    # Get selected feature names
    feature_names = np.array(vectorizer.get_feature_names_out())
    selected_features = feature_names[top_indices]

    return X_train_selected, top_indices, vectorizer


# Train MaxEnt model with class weights
def train_maxent_model(X_train, completions_train, X_val, completions_val):
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(completions_train), y=completions_train)
    class_weight_dict = dict(zip(np.unique(completions_train), class_weights))
    
    # Train a MaxEnt model (Logistic Regression with multinomial loss)
    model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight=class_weight_dict)
    model.fit(X_train, completions_train)

    # Validate the model
    val_predictions = model.predict(X_val)
    print("\nValidation Performance:")
    print(classification_report(completions_val, val_predictions))

    return model

# Perform 10-fold cross-validation with TF-IDF + IG
def cross_validate_maxent(prompts, completions, ngram_range=(2, 2), top_k=5000, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(prompts, completions)):
        print(f"\n=== Fold {fold + 1} ===")

        # Split data for the current fold
        prompts_train = [prompts[i] for i in train_index]
        completions_train = [completions[i] for i in train_index]
        prompts_val = [prompts[i] for i in val_index]
        completions_val = [completions[i] for i in val_index]

        X_train_selected, top_indices, vectorizer = select_features_df_ig(prompts_train, completions_train, ngram_range=(2, 2), top_k=5000)
        X_val = vectorizer.transform(prompts_val)[:, top_indices]

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(completions_train), y=completions_train)
        class_weight_dict = dict(zip(np.unique(completions_train), class_weights))

        # Train and evaluate the MaxEnt model
        model = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight=class_weight_dict)
        model.fit(X_train_selected, completions_train)
        val_predictions = model.predict(X_val)
        accuracy = accuracy_score(completions_val, val_predictions)
        fold_accuracies.append(accuracy)

        print(f"Fold Accuracy: {accuracy}")
        print(classification_report(completions_val, val_predictions))

    avg_accuracy = np.mean(fold_accuracies)
    print(f"\n=== Overall Cross-Validation Results ===")
    print(f"Average Accuracy: {avg_accuracy}")

# Evaluate the model on test data
def evaluate_model(model, vectorizer, prompts_test, completions_test, selected_features):
    X_test = vectorizer.transform(prompts_test)[:, selected_features]
    test_predictions = model.predict(X_test)
    print("\nTest Performance:")
    print(classification_report(completions_test, test_predictions))
    print("Accuracy:", accuracy_score(completions_test, test_predictions))

# Main function
def main(jsonl_file, ngram_range=(2, 2), top_k=5000):
    # Load the data
    print("Loading data...")
    prompts, completions = load_data(jsonl_file)

    # Split the dataset
    print("Splitting dataset into training, validation, and testing sets...")
    prompts_train, prompts_val, prompts_test, completions_train, completions_val, completions_test = split_dataset(
        prompts, completions, test_size=0.2, val_size=0.1
    )

    # Perform 10-fold cross-validation
    print("\nPerforming 10-fold cross-validation with DF + IG...")
    cross_validate_maxent(prompts_train + prompts_val, completions_train + completions_val, ngram_range=ngram_range, top_k=top_k)

    # Perform DF + IG feature selection for the final model
    print("\nPerforming DF + IG feature selection for the final model...")
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', binary=True, max_features=None)
    X_train = vectorizer.fit_transform(prompts_train)

    # Compute Information Gain (mutual information)
    ig_scores = mutual_info_classif(X_train, completions_train, discrete_features=True)

    # Select top-k features based on IG scores
    top_indices = np.argsort(ig_scores)[-top_k:]
    X_train_selected = X_train[:, top_indices]
    X_val = vectorizer.transform(prompts_val)[:, top_indices]

    # Train the MaxEnt model on the training set
    print("\nTraining the final MaxEnt model...")
    class_weights = compute_class_weight('balanced', classes=np.unique(completions_train), y=completions_train)
    class_weight_dict = dict(zip(np.unique(completions_train), class_weights))

    model = LogisticRegression(max_iter=1000, solver='lbfgs',  class_weight=class_weight_dict)
    model.fit(X_train_selected, completions_train)

    # Validate on the validation set
    val_predictions = model.predict(X_val)
    print("\nValidation Performance:")
    print(classification_report(completions_val, val_predictions))

    # Evaluate on the test dataset
    print("\nEvaluating on the test dataset...")
    X_test = vectorizer.transform(prompts_test)[:, top_indices]
    test_predictions = model.predict(X_test)
    print("\nTest Performance:")
    print(classification_report(completions_test, test_predictions))

# Run the workflow
if __name__ == "__main__":
    jsonl_file = "/Users/tosha/Documents/per-curiam/data/opinions.jsonl"  # Replace with your file path
    main(jsonl_file, ngram_range=(1, 1), top_k=5000)  # Adjust `ngram_range` and `top_k` as needed
