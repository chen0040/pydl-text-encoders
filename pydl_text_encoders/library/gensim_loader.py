from gensim.models import Word2Vec
import numpy as np

class GenSimWord2VecModel(object):

    def __init__(self):
        self.model = None

    def fit(self, sentences, embed_dim=None, window=None, min_count=None, workers=None):
        if window is None:
            window = 5
        if min_count is None:
            min_count = 5
        if workers is None:
            workers = 4
        sentence_input = []
        for sentence in sentences:
            sentence_input.append(sentence.strip().split(' '))

        self.model = Word2Vec(sentence_input, size=100, window=5, min_count=5, workers=4)
        self.model.init_sims(replace=True)

    def save_model(self, to_file):
        self.model.save(to_file)

    def load_model(self, from_file):
        self.model = Word2Vec.load(from_file)

    def encode_word(self, word):
        return self.model[word]

    def encode_docs(self, docs, max_allowed_doc_length=None):
        doc_count = len(docs)
        embedding_dim = self.model.
        X = np.zeros(shape=(doc_count, embedding_dim))
        max_len = 0
        for doc in docs:
            max_len = max(max_len, len(doc.split(' ')))
        if max_allowed_doc_length is not None:
            max_len = min(max_len, max_allowed_doc_length)
        for i in range(0, doc_count):
            doc = docs[i]
            words = [w.lower() for w in doc.split(' ')]
            length = min(max_len, len(words))
            E = np.zeros(shape=(self.embedding_dim, max_len))
            for j in range(length):
                word = words[j]
                try:
                    E[:, j] = self.word2em[word]
                except KeyError:
                    pass
            X[i, :] = np.sum(E, axis=1)

        return X

    def encode_doc(self, doc, max_allowed_doc_length=None):
        words = [w.lower() for w in doc.split(' ')]
        max_len = len(words)
        if max_allowed_doc_length is not None:
            max_len = min(len(words), max_allowed_doc_length)
        E = np.zeros(shape=(self.embedding_dim, max_len))
        X = np.zeros(shape=(self.embedding_dim, ))
        for j in range(max_len):
            word = words[j]
            try:
                E[:, j] = self.word2em[word]
            except KeyError:
                pass
        X[:] = np.sum(E, axis=1)
        return X

