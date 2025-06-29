# utils/vocab.py
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def __len__(self):
        return len(self.word2idx)

    def tokenizer(self, text):
        return text.lower().strip().split()

    def build_vocab(self, sentences):
        frequencies = Counter()
        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, text):
        return [
            self.word2idx.get(token, self.word2idx["<UNK>"])
            for token in self.tokenizer(text)
        ]
