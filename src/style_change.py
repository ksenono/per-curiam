import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import en_core_web_sm
# nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")

# Sample Dataset: Chronological Texts from One Author
data = [
    {"date": "2021-01", "text": "This is the first writing sample. It has short sentences."},
    {"date": "2021-06", "text": "Over time, the author's sentences grew longer and more complex, didn't they?"},
    {"date": "2022-01", "text": "A shift in punctuation: commas, dashes—more variation!"},
    {"date": "2022-06", "text": "Shorter again. More function words, less punctuation."},
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')  # Ensure texts are sorted chronologically

X = df["text"]
dates = df["date"]
y = df["author"]



# 1. Custom Stylometric Feature Extractor
class StylometricFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            doc = nlp(text)
            num_tokens = len(doc)
            num_sentences = len(list(doc.sents))
            avg_word_length = np.mean([len(token.text) for token in doc if token.is_alpha])
            num_punctuation = sum([1 for token in doc if token.is_punct])
            avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
            features.append([avg_word_length, num_punctuation, avg_sentence_length])
        return np.array(features)

# 2. Function Word Frequency Extractor
function_words = [
    "the", "is", "at", "which", "on", "and", "to", "with", "a", "for", "in", "it", "of", "that", "this", "by"
]

class FunctionWordFrequency(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        vectorizer = CountVectorizer(vocabulary=function_words)
        return vectorizer.fit_transform(X).toarray()

# 3. POS n-grams Extractor
class POSNGrams(BaseEstimator, TransformerMixin):
    def __init__(self, n=2):
        self.n = n
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        pos_sequences = []
        for text in X:
            doc = nlp(text)
            pos_tags = [token.pos_ for token in doc]
            ngrams = [" ".join(pos_tags[i:i+self.n]) for i in range(len(pos_tags)-self.n+1)]
            pos_sequences.append(" ".join(ngrams))
        
        # Vectorize POS n-grams
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(pos_sequences).toarray()

# 4. Combine All Features into a Pipeline
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('stylometric', StylometricFeatures()),
        ('function_words', FunctionWordFrequency()),
        ('pos_ngrams', POSNGrams(n=2)),
    ])),
    ('classifier', LogisticRegression())
])

# 5. Train-Test Split and Model Training

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# 6. Evaluation
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))




# # Combine Features
# feature_extractor = FeatureUnion([
#     ('stylometric', StylometricFeatures()),
#     ('function_words', FunctionWordFrequency()),
#     ('pos_ngrams', POSNGrams(n=2)),
# ])

# # Extract Features
# features = feature_extractor.fit_transform(X)

# # Convert to DataFrame for Analysis
# feature_names = ['avg_word_length', 'num_punctuation', 'avg_sentence_length'] + function_words + ['pos_ngram1', 'pos_ngram2']
# features_df = pd.DataFrame(features[:, :len(feature_names)], columns=feature_names)
# features_df['date'] = dates

# # 4. Plot Feature Changes Over Time
# def plot_feature_trends(df, features_to_plot):
#     plt.figure(figsize=(12, 8))
#     for feature in features_to_plot:
#         plt.plot(df['date'], df[feature], label=feature, marker='o')
#     plt.xlabel("Date")
#     plt.ylabel("Feature Value")
#     plt.title("Changes in Writing Style Over Time")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Plot Key Stylometric Features
# plot_feature_trends(features_df, ['avg_word_length', 'num_punctuation', 'avg_sentence_length'])

# # Plot Function Word Frequency
# plot_feature_trends(features_df, ['the', 'and', 'to', 'in'])


