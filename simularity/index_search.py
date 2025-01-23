from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def find_best_match(query, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_index = similarities.argmax()  # Get the index of the highest similarity score
    best_score = similarities[0, best_index]
    return best_score, best_index

def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return vectorizer, tfidf_matrix

