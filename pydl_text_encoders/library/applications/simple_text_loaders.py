def load_umich_sentiment_train_data(data_file_path):
    result = []
    with open(data_file_path, mode='rt', encoding='utf8') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            result.append((label, sentence))

    return result


class SentimentTrainDataSet(object):
    def __init__(self):
        self.data = None

    def load_umich(self, data_file_path):
        self.data = load_umich_sentiment_train_data(data_file_path)

    def get_docs(self):
        docs = []
        for _, doc in self.data:
            docs.append(doc)
        return docs

    def get_labels(self):
        labels = []
        for label, _ in self.data:
            labels.append(label)
        return labels
