from pydl_text_encoders.library.glove_loader import GloveModel
from pydl_text_encoders.library.applications.simple_text_loaders import SentimentTrainDataSet


def main():
    glove_model = GloveModel()
    glove_model.load('../very_large_data')

    print('current encoding is: ', glove_model.embedding_dim)
    print(glove_model.encode_word('文'))
    print(glove_model.encode_word('text'))

    data_set = SentimentTrainDataSet()
    data_set.load_umich('../data/umich-sentiment-train.txt')

    docs = data_set.get_docs()

    for doc in docs[0:10]:
        print('Origin: ', doc)
        print('Encoded: ', glove_model.encode_doc(doc))


if __name__ == '__main__':
    main()
