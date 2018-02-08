from pydl_text_encoders.library.applications.simple_text_loaders import SentimentTrainDataSet
from pydl_text_encoders.library.gensim_loader import GenSimWord2VecModel


def main():
    data_dir_path = './data'

    sentences = []
    # Import `umich-sentiment-train.txt`
    with open(data_dir_path + '/umich-sentiment-train.txt', mode='rt', encoding='utf8') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            sentences.append(sentence)

    gs = GenSimWord2VecModel()
    gs.fit(sentences)
    # gs.load_google_news_vectors('./very_large_data')

    print('current encoding is: ', gs.size())
    print(gs.encode_word('文'))
    print(gs.encode_word('text'))

    data_set = SentimentTrainDataSet()
    data_set.load_umich('./data/umich-sentiment-train.txt')

    docs = data_set.get_docs()

    for doc in docs[0:10]:
        print('Origin: ', doc)
        print('Encoded: ', gs.encode_doc(doc))


if __name__ == '__main__':
    main()
