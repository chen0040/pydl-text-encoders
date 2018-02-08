from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier

from pydl_text_encoders.library.gensim_loader import GenSimWord2VecModel
from pydl_text_encoders.library.plot_utils import plot_confusion_matrix


def main():
    data_dir_path = '../../data'

    # Import `umich-sentiment-train.txt`
    df = pd.read_csv(data_dir_path + "/umich-sentiment-train.txt", sep='\t', header=None, usecols=[0, 1],
                     encoding='utf8')

    print(df.head())

    Y = df[0].as_matrix()
    X = df[1].as_matrix()

    encoder = GenSimWord2VecModel()
    encoder.load_google_news_vectors('../../very_large_data')
    X = encoder.encode_docs(X)

    # Make training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 2), random_state=42)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1])


if __name__ == '__main__':
    main()
