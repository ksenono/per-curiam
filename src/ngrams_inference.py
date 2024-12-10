import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]

# Define the neural network model architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the data from JSONL file
def load_data(jsonl_file):
    prompts, names = [], []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            prompts.append(data['text'])
            if 'filename' in names:
                names.append(data['filename'])
            else:
                names.append('')
    return prompts, names

# Load and set up the vectorizer 
def setup_vectorizer(train_prompts, ngram_range=(2, 2), top_k=5000):
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        binary=True,
        min_df=1,
        max_df=1.0
    )
    X_train = vectorizer.fit_transform(train_prompts)
    top_indices = np.argsort(X_train.sum(axis=0)).A1[-top_k:]
    return vectorizer, top_indices

# Main process for handling inference
def perform_inference(jsonl_file, model_weights_path, ngram_size, num_classes=8, top_k=5000):
    # Load and process the dataset
    prompts, names = load_data(jsonl_file)
    vectorizer_params = {'ngram_range': (ngram_size, ngram_size)}
    vectorizer, top_indices = setup_vectorizer(prompts, **vectorizer_params)
    
    # Vectorize input prompts
    X_features = vectorizer.transform(prompts)[:, top_indices].toarray()
    
    # Verify the number of features matches the expected input dimension
    assert X_features.shape[1] == top_k, f"Expected {top_k} features but got {X_features.shape[1]}"
    
    # Prepare DataLoader
    dataset = TextDataset(torch.tensor(X_features, dtype=torch.float32))
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Initialize and load the model
    model = SimpleNN(input_dim=top_k, output_dim=num_classes).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # Evaluate model on data
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    return [{'pred': pred, 'name': name} for pred, name in zip(predictions, names)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-gram model inference using PyTorch')
    parser.add_argument('jsonl', type=str, help='The JSONL input file')
    parser.add_argument('-w', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('-n', type=int, required=True, help='Size of the n-grams to use')
    parser.add_argument('-c', type=int, default=8, help='Number of output classes')
    parser.add_argument('-r', type=str, default='./results.json', help='File to save the results')
    
    args = parser.parse_args()
    
    # Perform inference and convert predictions to a list of Python integers
    output = perform_inference(args.jsonl, args.w, args.n, num_classes=args.c)
    predictions = [{'pred': int(entry['pred']), 'name': entry['name']} for entry in output]  # Convert from numpy int64 to Python int
    
    # Save predictions to an output file
    with open(args.r, 'w') as f:
        json.dump(predictions, f)

    print(f"Results saved to {args.r}.")