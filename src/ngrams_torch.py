import json
import numpy as np
import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

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

def load_data(jsonl_file):
    prompts, completions = [], []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            prompts.append(data['text'])
            completions.append(data['author'])
    return prompts, completions

def normalize_by_author(prompts, labels, max_samples_per_author=100):
    data = list(zip(prompts, labels))
    normalized_data = []
    counter = Counter(labels)

    for author in set(labels):
        author_data = [d for d in data if d[1] == author]
        normalized_data.extend(author_data[:max_samples_per_author])  # Limit to max samples

    normalized_prompts, normalized_labels = zip(*normalized_data)
    return list(normalized_prompts), list(normalized_labels)


def select_features_df_ig(prompts_train, completions_train, ngram_range=(2, 2), top_k=5000):
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        binary=True,
        min_df=1,  # Lower this to ensure almost all terms are included initially
        max_df=1.0 # Allow for all terms if needed
    )
    X_train = vectorizer.fit_transform(prompts_train)
    ig_scores = mutual_info_classif(X_train, completions_train, discrete_features=True)
    top_indices = np.argsort(ig_scores)[-top_k:]
    X_train_selected = X_train[:, top_indices]

    feature_names = np.array(vectorizer.get_feature_names_out())
    selected_features = feature_names[top_indices]
    selected_scores = ig_scores[top_indices]

    return X_train_selected, top_indices, selected_features, selected_scores, vectorizer

def train_and_evaluate(model, train_loader, val_loader, class_weights, num_epochs=10, lr=0.01):
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset) * 100
        print(f'Epoch {epoch + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

def main(jsonl_file, ngram_range=(2, 2), top_k=5000, save_dir='./model_weights.pth', features_file='./selected_features.txt'):
    print("Loading data...")
    prompts, completions = load_data(jsonl_file)

    label_encoder = LabelEncoder()
    completions_encoded = label_encoder.fit_transform(completions)

    split_ratio = 0.8
    split_point = int(split_ratio * len(prompts))
    prompts_train, prompts_val = prompts[:split_point], prompts[split_point:]
    completions_train, completions_val = completions_encoded[:split_point], completions_encoded[split_point:]

    prompts_train, completions_train = normalize_by_author(prompts_train, completions_train)
    print("Selecting features...")
    X_train_selected, top_indices, selected_features, selected_scores, vectorizer = select_features_df_ig(prompts_train, completions_train, ngram_range=ngram_range, top_k=top_k)
    X_train_selected = X_train_selected.toarray()

    print(f"Saving selected features and their weights to {features_file}...")
    with open(features_file, 'w') as f:
        for feature, score in zip(selected_features, selected_scores):
            f.write(f"{feature}\t{score}\n")

    X_val = vectorizer.transform(prompts_val)[:, top_indices].toarray()

    train_dataset = TextDataset(torch.tensor(X_train_selected, dtype=torch.float32), torch.tensor(completions_train, dtype=torch.long))
    val_dataset = TextDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(completions_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    class_weights = compute_class_weight('balanced', classes=np.unique(completions_train), y=completions_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = SimpleNN(input_dim=X_train_selected.shape[1], output_dim=len(label_encoder.classes_)).to(device)
    print("\nTraining the model...")
    train_and_evaluate(model, train_loader, val_loader, class_weights)

    print(f"Saving model weights to {save_dir}...")
    torch.save(model.state_dict(), save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='N-gram model training using PyTorch')
    parser.add_argument('jsonl', type=str, help='The location of the file')
    parser.add_argument('-n', type=int, required=True, help='context size')
    parser.add_argument('-w', type=str, help='Directory to save the model weights')
    
    args = parser.parse_args()
    main(args.jsonl, (args.n, args.n), save_dir=args.w + '.pth', features_file=args.w + '.txt')