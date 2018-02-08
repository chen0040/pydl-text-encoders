import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from pydl_text_encoders.library.glove_loader import GloveModel

BATCH_SIZE = 16
NUM_EPOCHS = 20


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def main():
    np.random.seed(42)
    data_dir_path = '../data'

    Xdata = []
    Ydata = []
    # Import `umich-sentiment-train.txt`
    with open(data_dir_path + '/umich-sentiment-train.txt', mode='rt', encoding='utf8') as file:
        for line in file:
            label, sentence = line.strip().split('\t')
            Xdata.append(sentence)
            Ydata.append(int(label))

    glove_model = GloveModel()
    glove_model.load('../very_large_data')
    Xdata = glove_model.encode_docs(Xdata)

    print(Xdata.shape)

    Ydata = np.array([[1 - y, y] for y in Ydata])

    print(Ydata.shape)

    # Make training and test sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.2, random_state=42)

    # Layer's sizes
    x_size = Xtrain.shape[1]  # Number of inputs
    h_size = 64  # Number of hidden nodes
    y_size = 2

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(NUM_EPOCHS):
        # Train with each example
        for i in range(len(Xtrain) // BATCH_SIZE):
            sess.run(updates, feed_dict={X: Xtrain[i: i + BATCH_SIZE], y: Ytrain[i: i + BATCH_SIZE]})

        train_accuracy = np.mean(np.argmax(Ytrain, axis=1) ==
                                 sess.run(predict, feed_dict={X: Xtrain, y: Ytrain}))
        test_accuracy = np.mean(np.argmax(Ytest, axis=1) ==
                                sess.run(predict, feed_dict={X: Xtest, y: Ytest}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()


if __name__ == '__main__':
    main()
