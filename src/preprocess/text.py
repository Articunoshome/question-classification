"""
Contains the classes and functions for preprocessing text data
"""
import re
from collections import Counter

import numpy as np
import torch

_patterns = [r'\'',
             r'\"',
             r'\.',
             r'<br \/>',
             r',',
             r'\(',
             r'\)',
             r'\!',
             r'\?',
             r'\;',
             r'\:',
             r'\s+']
_replacements = [' \'  ',
                 '',
                 ' . ',
                 ' ',
                 ' , ',
                 ' ( ',
                 ' ) ',
                 ' ! ',
                 ' ? ',
                 ' ',
                 ' ',
                 ' ']

_patterns_dict = list((re.compile(p), r)
                      for p, r in zip(_patterns, _replacements))


class VocabBuilder():
    """
    Class to build the text vocabulary
    """

    def __init__(self, lowercase=True):
        self.stoi = dict()
        self.itos = []
        self.embedding = dict()
        self.itov = []
        self.tokens = []
        self.unk_token = "#UNK#"
        self.pad_token = "<pad>"
        self.lower = lowercase

    def _basic_english_normalize(self, line):
        if self.lower:
            line = line.lower()
        for pattern_re, replaced_str in _patterns_dict:
            line = pattern_re.sub(replaced_str, line)
        return line.split()

    def find_n_gram(self, word, n=3):
        return [word[i:i+n] for i in range(len(word)-n+1)]

    def find_average_vector(self, word, n=3):
        _w_n = len(word)
        vecs = []
        if _w_n <= 3:
            ngrams = self.find_n_gram(word, _w_n-1)
        else:
            ngrams = self.find_n_gram(word, n)
        vecs = np.array(
            list(filter(lambda x: x != None, map(self.embedding.get, ngrams))))
        if vecs.shape[0]:
            return np.average(vecs, axis=0).tolist()
        else:
            return self.embedding.get(self.unk_token)

    def get_token_embbedding(self, token):
        vec = self.embedding.get(token)
        if vec:
            return vec
        else:
            if len(token) > 1:
                return self.find_average_vector(token)
            else:
                return self.embedding[self.unk_token]

    def build_vocab_from_iterator(self, iterator, stop_words=[], min_freq=1, embedding=False):
        counter = Counter()
        for _item in iterator:
            counter.update(_item)
        sorted_by_freq_tuples = sorted(
            counter.items(), key=lambda x: x[1], reverse=True)
        if embedding:
            for token, freq in sorted_by_freq_tuples:
                if freq >= min_freq and token not in stop_words:
                    self.itos.append(token)
                    self.itov.append(self.get_token_embbedding(token))
        else:
            self.itos.extend(
                [token for token, freq in sorted_by_freq_tuples
                 if freq >= min_freq and token not in stop_words])

        if self.unk_token not in self.itos:
            self.itos.insert(0, self.unk_token)
            self.itov.insert(0, self.embedding.get(self.unk_token))
            self.itos.insert(0, self.pad_token)
            self.itov.insert(0, [0.0]*300)
        self.stoi.update(zip(self.itos, range(len(self.itos))))

    def build_vocab(self, text_list, min_freq=1,
                    emb_file=None, unk_token="#UNK#", pad_token="<pad>", lower=True):
        stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                      "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                      "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
                      "their", "theirs", "themselves", "this", "that", "these", "those",
                      "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                      "while", "of", "at", "by", "for", "with", "about", "against",
                      "between", "into", "through", "during", "before", "after", "above",
                      "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                      "under", "again", "further", "then", "once", "here", "there", "all", "any",
                      "both", "each", "few", "more", "most", "other", "some", "such", "no",
                      "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
                      "just", "don", "now", "'", ",", "''", "``", "'s", "!", "(", ")", "-", "[",
                      "]", "{", "}", ";", ":", "?", "@", "*", "_", "~", "."]

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lower = lower
        s_tokens_list = list(map(self._basic_english_normalize, text_list))
        if emb_file:
            with open(emb_file) as fp:
                for line in fp.readlines():
                    values = line.split()
                    self.embedding[values[0]] = list(map(float, values[1:]))
            self.build_vocab_from_iterator(
                s_tokens_list, stop_words, min_freq, True)
        else:
            self.build_vocab_from_iterator(
                s_tokens_list, stop_words, min_freq, False)
        tokens = [torch.tensor(list(filter(lambda x: x != None, map(
            self.stoi.get, s_tokens)))) for s_tokens in s_tokens_list]
        offsets = torch.LongTensor(list(map(len, tokens)))
        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=self.stoi[self.pad_token])
        return tokens, offsets

    def convert_sentences_to_encoding(self, sentences):
        """
        Given list of sentences function returns list of encoded tokens
        """
        encoded = []
        for sentence in sentences:
            tokenized = self._basic_english_normalize(
                sentence)  # tokenize the sentence
            indexed = [self.stoi.get(token) if self.stoi.get(
                token) else self.stoi.get(self.unk_token) for token in tokenized]
            encoded.append(indexed)
        return encoded

    def convert_encodings_to_sentences(self, encodings):
        """
        Given a list of encoded sentence tokens function will convert it
        back to list of sentences
        """
        return [' '.join(map(lambda x: self.itos[x], encoding)) for encoding in encodings]
