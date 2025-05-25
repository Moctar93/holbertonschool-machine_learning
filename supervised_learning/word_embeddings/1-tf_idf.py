#!/usr/bin/env python3
"""
    TF-IDF
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    preprocessed_sentences = [
            re.sub(r"\b(\w+)'s\b", r"\1",
                   sentence.lower()) for sentence in sentences
            ]

    if vocab is None:
        list_words = []
        for sentence in preprocessed_sentences:
            words = re.findall(r'\w+', sentence)
            list_words.extend(words)
        vocab = sorted(set(list_words))

    tfidf_vect = TfidfVectorizer(vocabulary=vocab)

    tfidf_matrix = tfidf_vect.fit_transform(sentences)

    features = tfidf_vect.get_feature_names_out()

    return tfidf_matrix.toarray(), features
