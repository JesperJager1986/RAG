from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
import nltk
from nltk.corpus import wordnet
from collections import Counter
import numpy as np

nltk.download('wordnet')

def find_best_match_keyword_search(query: str, db_records: list[str]):

    query_keywords = set(query.lower().split())  # Convert query to a set of unique keywords

    record_keywords = list(map(lambda x: set(x.lower().split()), db_records))

    common_keywords = list(map(lambda x: x.intersection(query_keywords), record_keywords))

    scores = np.array(list(map(lambda x: len(x), common_keywords)))

    return scores[np.argmax(scores)], db_records[np.argmax(scores)]

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True,   # Apply sublinear TF scaling
        analyzer='word'      # You could also experiment with 'char' or 'char_wb' for character-level features
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def preprocess_text(text):
    # python - m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    lemmatized_words = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]

    return lemmatized_words

def expand_with_synonyms(words):
    expanded_words = words.copy()
    for word in words:
        expanded_words.extend(get_synonyms(word))
    return expanded_words

def calculate_enhanced_similarity(text1: str, text2: str):
    # Preprocess and tokenize texts
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Expand with synonyms
    words1_expanded = expand_with_synonyms(words1)
    words2_expanded = expand_with_synonyms(words2)

    # Count word frequencies
    freq1 = Counter(words1_expanded)
    freq2 = Counter(words2_expanded)

    # Create a set of all unique words
    unique_words = set(freq1.keys()).union(set(freq2.keys()))

    # Create frequency vectors
    vector1 = [freq1[word] for word in unique_words]
    vector2 = [freq2[word] for word in unique_words]

    # Convert lists to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate cosine similarity
    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return cosine_similarity