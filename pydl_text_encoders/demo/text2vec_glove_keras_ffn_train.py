from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import np_utils
import numpy as np

from pydl_text_encoders.library.glove_loader import GloveModel
from pydl_text_encoders.library.text_utils import remove_infrequent_words

BATCH_SIZE = 64
NUM_EPOCHS = 20


def main():
    np.random.seed(42)
    data_dir_path = './data'

    X = []
    Y = []
    # Import `umich-sentiment-train.txt`
    with open(data_dir_path + '/umich-sentiment-train.txt', mode='rt', encoding='utf8') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            X.append(sentence)
            Y.append(label)

    # X = remove_infrequent_words(X)

    glove_model = GloveModel()
    glove_model.load('./very_large_data')
    X = glove_model.encode_docs(X)

    print(X.shape)

    Y = np_utils.to_categorical(Y, 2)

    # Make training and test sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=Xtrain.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=[Xtest, Ytest])

    score = model.evaluate(x=Xtrain, y=Ytrain, verbose=1)

    print('score: ', score[0])
    print('accuracy: ', score[1])


if __name__ == '__main__':
    main()