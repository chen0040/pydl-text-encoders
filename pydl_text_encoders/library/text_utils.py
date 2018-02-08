from collections import Counter


def get_most_frequent_words(X, max_vocab_size=None):
    if max_vocab_size is None:
        max_vocab_size = 5000

    counter = Counter()
    for sentence in X:
        tokens = [w.lower() for w in sentence.split(' ')]
        for token in tokens:
            counter[token] += 1
    return set([word for word, idx in counter.most_common(max_vocab_size)])

