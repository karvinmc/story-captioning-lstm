from collections import Counter


def simple_tokenize(text):
    return text.lower().split()


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocab(self, sentences):
        freq = Counter()
        idx = 4
        for sentence in sentences:
            for word in simple_tokenize(sentence):
                freq[word] += 1
                if freq[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        return [
            self.stoi.get(word, self.stoi["<UNK>"]) for word in simple_tokenize(text)
        ]
