from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import os
import urllib.request

from pydl_text_encoders.library.download_utils import reporthook


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
        if embed_dim is None:
            embed_dim = 100
        sentence_input = []
        for sentence in sentences:
            sentence_input.append(sentence.strip().split(' '))

        self.model = Word2Vec(sentence_input, size=embed_dim, window=window, min_count=min_count, workers=workers)
        self.model.init_sims(replace=True)

    def load_google_news_vectors(self, data_dir_path):
        word2vec_model = data_dir_path + '/GoogleNews-vectors-negative300.bin.gz'
        if not os.path.exists(word2vec_model):
            print('word2vec_model file not found locally, downloading from internet')
            url_link = 'https://www.dropbox.com/s/i6vkmpr8ge4dce2/GoogleNews-vectors-negative300.bin.gz?dl=1'
            urllib.request.urlretrieve(url=url_link, filename=word2vec_model,
                                       reporthook=reporthook)
        self.model = KeyedVectors.load_word2vec_format(word2vec_model, binary=True)

    def save_model(self, to_file):
        self.model.save(to_file)

    def load_model(self, from_file):
        self.model = Word2Vec.load(from_file)

    def encode_word(self, word):
        result = np.zeros(shape=(self.size(), ))
        try:
            result = self.model[word]
        except KeyError:
            pass
        return result

    def size(self):
        return self.model.wv.vector_size

    def encode_docs(self, docs, max_allowed_doc_length=None):
        doc_count = len(docs)
        embedding_dim = self.size()
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
            E = np.zeros(shape=(embedding_dim, max_len))
            for j in range(length):
                word = words[j]
                try:
                    E[:, j] = self.model[word]
                except KeyError:
                    pass
            X[i, :] = np.sum(E, axis=1)

        return X

    def encode_doc(self, doc, max_allowed_doc_length=None):
        words = [w.lower() for w in doc.split(' ')]
        max_len = len(words)
        embedding_dim = self.size()
        if max_allowed_doc_length is not None:
            max_len = min(len(words), max_allowed_doc_length)
        E = np.zeros(shape=(embedding_dim, max_len))
        X = np.zeros(shape=(embedding_dim,))
        for j in range(max_len):
            word = words[j]
            try:
                E[:, j] = self.model[word]
            except KeyError:
                pass
        X[:] = np.sum(E, axis=1)
        return X
