import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, prompts, labels):
        self.prompts = prompts
        self.labels = labels
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data from the JSONL file
def load_data(jsonl_file):
    prompts, completions = [], []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            prompts.append(data['text'])
            completions.append(data['author'])
    return prompts, completions

# Select features using Document Frequency (DF) and Information Gain (IG)
def select_features_df_ig(prompts_train, completions_train, ngram_range=(2, 2), top_k=5000):
    vectorizer = CountVectorizer(ngram_range=ngram_range, binary=True, min_df=5, max_df=0.9)
    X_train = vectorizer.fit_transform(prompts_train)
    ig_scores = mutual_info_classif(X_train, completions_train, discrete_features=True)
    top_indices = np.argsort(ig_scores)[-top_k:]
    X_train_selected = X_train[:, top_indices]

    feature_names = np.array(vectorizer.get_feature_names_out())
    selected_features = feature_names[top_indices]

    return X_train_selected, top_indices, selected_features, vectorizer

# Load model weights and evaluate
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
    
    val_accuracy = correct / len(val_loader.dataset) * 100
    return val_accuracy

def process_folder(folder_path, model_weights_path, ngram_range=(2, 2), top_k=5000, results_file='./results.json'):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path}...")

            # Load the data
            prompts, completions = load_data(file_path)

            # Encode labels
            label_encoder = LabelEncoder()
            completions_encoded = label_encoder.fit_transform(completions)

            # Split the dataset into training and validation
            split_ratio = 0.8
            split_point = int(split_ratio * len(prompts))
            prompts_train, prompts_val = prompts[:split_point], prompts[split_point:]
            completions_train, completions_val = completions_encoded[:split_point], completions_encoded[split_point:]

            # Perform feature selection
            X_train_selected, top_indices, selected_features, vectorizer = select_features_df_ig(
                prompts_train, completions_train, ngram_range=ngram_range, top_k=top_k)
            X_train_selected = X_train_selected.toarray()

            # Transform validation data
            X_val = vectorizer.transform(prompts_val)[:, top_indices].toarray()

            # Convert data to PyTorch datasets
            val_dataset = TextDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(completions_val, dtype=torch.long))

            # Create data loaders
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            # Initialize model and load weights
            model = SimpleNN(input_dim=X_train_selected.shape[1], output_dim=len(label_encoder.classes_)).to(device)
            model.load_state_dict(torch.load(model_weights_path))

            # Evaluate the model
            val_accuracy = evaluate_model(model, val_loader)

            # Add result to the list
            results.append({
                'file': filename,
                'val_accuracy': val_accuracy,
                'num_classes': len(label_encoder.classes_)
            })

    # Save results to a JSON file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model on all JSONL files in a folder using existing weights.')
    parser.add_argument('folder', type=str, help='The folder containing JSONL files.')
    parser.add_argument('-w', type=str, required=True, help='Path to the file containing the model weights')
    parser.add_argument('-n', type=int, required=True, help='context size')
    parser.add_argument('-s', type=str, default='./results.json', help='File to save the results')
    
    args = parser.parse_args()
    process_folder(args.folder, args.w, (args.n, args.n), results_file=args.s)