from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import json,re
word_re = re.compile(r"[\w']+")

# import training, testing data, embedding and dictionary
train_data = []
test_data = []
with open('train.txt', 'r') as f, open('test.txt', 'r') as t:
    for line in f:
        train_data.append(line[:-1])
    for line in t:
        test_data.append(line[:-1])

with open('dictionary.txt', 'r') as f:
    dictionary = json.load(f)

train_temp = pd.read_csv('train.csv')['label'].values
test_temp = pd.read_csv('test.csv')['label'].values
embed = pd.read_csv('final_embedding.csv').drop('Unnamed: 0', axis=1).values

train_label = []
test_label = []
for i in train_temp:
    if i == 0:
        train_label.append([0,1])
    else: train_label.append([1,0])
for i in test_temp:
    if i == 0:
        test_label.append([0, 1])
    else: test_label.append([1, 0])

# return batch in shape [batch_size, sentence_len, embedding_len]
def generate_batch(data, label, batch_size, indeces):
    batch_x = []
    batch_y = []
    batch_l = []
    embedLen = embed.shape[1]
    maxlen = -1
    for i in range(batch_size):
        sentence = []
        length = 0
        for word in word_re.findall(data[i]):
            if word.isdigit() or word == "'": continue
            if word[0] == "'": word = word[1:]
            if word[-1] == "'": word = word[:-2]
            sentence.append(embed[dictionary.get(word.lower(), 0)])
            length += 1
        batch_x.append(sentence)
        batch_y.append(label[indeces])
        indeces = (indeces + 1) % len(data)
        batch_l.append(length)
        if length > maxlen: maxlen = length

    # padding
    emptyword = np.zeros(embedLen)
    for i in range(batch_size):
        for j in range(maxlen - len(batch_x[i])):
            batch_x[i].append(emptyword)

    return batch_x, batch_y, batch_l, indeces

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 100
hidden_layer = 1024
classes = 2
embeddingLen = 128

# tf Graph input
x = tf.placeholder("float", [None, None, embeddingLen], name='Inputs')
y = tf.placeholder("float", [None, classes], name='outputs')
seqlen = tf.placeholder(tf.int32, [None])
# Define weights
w = tf.Variable(tf.random_normal([hidden_layer, classes]), name='weights')
b = tf.Variable(tf.random_normal([classes]), name='bias')

def RNN(x, seqlen, w, b):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, x, seqlen, dtype=tf.float32)
    batch_size = tf.shape(outputs)[0]
    dataMLen = tf.shape(x)[1]
    index = tf.range(0, batch_size) * dataMLen + (seqlen - 1)
    output = tf.gather(tf.reshape(outputs, [-1, hidden_layer]), index)
    output = tf.nn.softmax(tf.matmul(output, w) + b)
    return output

# set lose function
pred = RNN(x, seqlen, w, b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# compute the accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

# run network
with tf.Session() as sess:
    sess.run(init)
    indeces = 0
    for i in range(training_iters):
        batch_x, batch_y, batch_l, indeces = generate_batch(train_data, train_label, batch_size=batch_size, indeces=indeces)

        output = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_l})
        if i % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_l})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_l})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    indeces = 0
    test_pret = []
    for i in range(0, len(test_data), batch_size):
        if len(test_data) - indeces < 128: batch_size = len(test_data) - indeces
        batch_x, batch_y, batch_l, indeces = generate_batch(test_data, test_label, batch_size=batch_size, indeces=indeces)
        test_pret.extend(sess.run(pred, feed_dict={x: batch_x, y: batch_y, seqlen: batch_l}).tolist())
    test_pret = np.array(test_pret)
    accu = np.equal(test_pret.argmax(1), np.array(test_label).argmax(1)).astype(np.int32)
    accu = accu.sum() / len(accu)
    print("Test accuracy: ", accu)
