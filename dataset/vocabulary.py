import json
import os

class Vocabulary(object):
    """
    A simple Vocabulary class which maintains a mapping between words and
    integer tokens.
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, word_counts_path, min_count = 5):
        if not os.path.exists(word_counts_path):
            raise FileNotFoundError(
                f"Word counts do not exist at {word_counts_path}"
            )

        with open(word_counts_path, "r") as word_counts_file:
            word_counts = json.load(word_counts_file)

            # form a list of (word, count) tuples and apply min_count threshold
            word_counts = [(word, count) for word, count in word_counts.items() if count >= min_count]
            
            # sort in descending order of word counts
            word_counts = sorted(word_counts, key=lambda wc: -wc[1])
            words = [w[0] for w in word_counts]

        self.word2index = {}
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        for index, word in enumerate(words):
            self.word2index[word] = index + 4

        self.index2word = {
            index: word for word, index in self.word2index.items()
        }

    def to_indices(self, words):
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_words(self, indices):
        return [self.index2word.get(index, self.UNK_TOKEN) for index in indices]

    def __len__(self):
        return len(self.index2word)
