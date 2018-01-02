""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons
n_input = 128 # MNIST data input (img shape: 28*28)
n_out = 128 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_out])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_out]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_out]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def read_embeddings(filename):
    embedding = dict()
    with open(filename, 'r') as f_handler:
        for ln in f_handler:
            ln = ln.strip()
            if ln:
                elems = ln.split()
                if len(elems)==2:
                    continue
                embedding[elems[0]] = elems[1:]
    return embedding

def read_identity_labels(filename):
    lbs = list()
    with open(filename, 'r') as lb_f_handler:
        for ln in lb_f_handler:
            ln = ln.strip()
            if ln:
                labels = ln.split()
                dou_id = labels[0]
                w_id = labels[1]
                lbs.append([dou_id, w_id])
    return lbs

def get_batches(labels, embeddings_src, embeddings_obj, batch_size):
    idx = 0
    while idx<len(labels):
        batch_x = list()
        batch_y = list()
        for i in range(min(batch_size, len(labels)-idx)):
            lb_src,lb_obj = labels[idx]
            batch_x.append(embeddings_src[lb_src])
            batch_y.append(embeddings_obj[lb_obj])
            idx += 1
        yield np.array(batch_x, np.float32), np.array(batch_y, np.float32)

def model_test(test_labels, embeddings_src, embeddings_obj, res_file):
    batches_test = get_batches(test_labels, embeddings_src, embeddings_obj, batch_size)
    obj_keys = embeddings_obj.keys()

    res_handler = open('res_file', 'w')
    sample_size = 9
    mrr = .0
    for label_src, label_obj in test_labels:
        neg_dist = tf.zeros(sample_size)
        src_vec = tf.placeholder('float', [n_input])
        obj_vec = tf.placeholder('float', [n_out])
        dist = tf.norm(src_vec-obj_vec)
        for i in range(sample_size):
            rand_w_id = obj_keys[np.random.randint(0,len(obj_keys))]
            while label_obj==rand_w_id:
                rand_w_id = obj_keys[np.random.randint(0,len(obj_keys))]
            neg_dist[i] = dist.eval({src_vec:embeddings_src[label_src], obj_vec:embeddings_obj[rand_w_id]})
        tag_dist = dist.eval({src_vec:embeddings_src[label_src], obj_vec:embeddings_obj[label_obj]})
        pos = 1
        for i in range(sample_size):
            if tag_dist>neg_dist[i]:
                pos+=1
        cur_mrr = 1./pos
        mrr += cur_mrr
        res_handler.write('{},{}:{},{}'.format(label_src, label_obj, pos, cur_mrr))
    res_handler.write('overall:{}'.format(mrr/len(test_labels)))

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    embeddings_src = read_embeddings('/home/yangyaru/project/alignment/data/embedding/Douban_1st.txt')
    embeddings_obj = read_embeddings('/home/yangyaru/project/alignment/data/embedding/Weibo_1st.txt')
    train_labels = read_identity_labels('/home/yangyaru/project/alignment/data/train/multitrain/embedding_train.txt')
    test_labels = read_identity_labels('/home/yangyaru/project/alignment/data/train/multitrain/embedding_test.txt')

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        batches = get_batches(train_labels, embeddings_src, embeddings_obj, batch_size)
        # Loop over all batches
        cnt = 0
        for batch_x,batch_y in batches:
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c
            cnt += 1
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost/cnt))

        # Test model
        model_test(test_labels, embeddings_src, embeddings_obj, 'eval_res')
    print("Optimization Finished!")

