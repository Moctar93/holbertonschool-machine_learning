#!/usr/bin/env python3
"""
A function that creates a bag-of-words.
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    bag of words embedding.
    """
    word_set = set()
    processed_sentences = []

    for sentence in sentences:
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        processed_sentences.append(words)
        if vocab is None:
            word_set.update(words)

    if vocab is None:
        vocab = sorted(word_set)

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, np.array(vocab)
