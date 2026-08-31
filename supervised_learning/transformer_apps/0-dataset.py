#!/usr/bin/env python3
"""
Create the class Dataset that loads and preps a dataset for machine translation
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and preps a dataset
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset

        :param data: a tf.data.Dataset whose examples are formatted as a
            tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the English sentence

        :return: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_sentences = (
            pt.numpy().decode('utf-8') for pt, en in data
        )

        en_sentences = (
            en.numpy().decode('utf-8') for pt, en in data
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences,
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences,
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
```

