import nltk
from collections import Counter


def get_most_frequent_words(X, max_vocab_size=None):
    if max_vocab_size is None:
        max_vocab_size = 5000

    counter = Counter()
    for sentence in X:
        tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
        for token in tokens:
            counter[token] += 1
    return set([word for word, idx in counter.most_common(max_vocab_size)])


def remove_infrequent_words(X, max_vocab_size=None):
    if max_vocab_size is None:
        max_vocab_size = 5000

    result = []
    frequent_words = get_most_frequent_words(X, max_vocab_size)
    for i in range(0, len(X)):
        sentence = X[i]
        tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
        temp = []
        for token in tokens:
            if token in frequent_words:
                temp.append(token)
        result.append(' '.join(temp))
    print(result[0:20])
    return result
