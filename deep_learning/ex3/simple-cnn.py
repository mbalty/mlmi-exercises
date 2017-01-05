import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


def conv2d(x, conv_size, input_channels, output_channels):
    with tf.name_scope('conv'):
        weights = tf.Variable(tf.random_normal([conv_size, conv_size, input_channels, output_channels]))
        bias = tf.Variable(tf.random_normal([output_channels]))
        return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME') + bias


def relu(x):
    with tf.name_scope('relu'):
        return tf.nn.relu(x)


def fully_connected(x, input_size, output_size, dropout_keep_prob=None):
    with tf.name_scope('fc'):
        weights = tf.Variable(tf.random_normal([input_size, output_size]))
        bias = tf.Variable(tf.random_normal([output_size]))
        out = tf.add(tf.matmul(x, weights), bias)

        if dropout_keep_prob:
            with tf.name_scope('dropout'):
                out = tf.nn.dropout(out, dropout_keep_prob)
        return out


def maxpool2d(x):
    with tf.name_scope('maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(x, conv_size, input_channels, output_channels):
    conv = conv2d(x, conv_size, input_channels, output_channels)
    conv = relu(conv)
    conv = maxpool2d(conv)
    return conv


def cnn_model(x, n_classes=10):
    # preprocess input so that it works with the digits lib
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    x = tf.image.grayscale_to_rgb(tf.image.resize_images(x, new_height=64, new_width=64))
    # x = tf.reshape(x, shape=[-1, 64, 64, 3])
    print('in:', x)


    # convolution layers
    data = x
    isz = 3
    osz = 32
    for i in range(4):
        conv = conv_layer(data, 3, isz, osz)
        isz = osz
        osz *= 2
        data = conv
        print('conv', i, conv)

    # flattening and fully connected layers
    flt_sz = 4 * 4 * 256
    fc = tf.reshape(data, [-1, flt_sz])
    fc = relu(fc)
    print('fc',0,fc)
    fc = fully_connected(fc, flt_sz, 1024)
    fc = relu(fc)
    print('fc', 1, fc)


    # out
    with tf.name_scope('output'):
        output = fully_connected(fc, 1024, n_classes)
    print('out', output)
    return output


def train_cnn(X, y, train=False, batch_size=128, hm_epochs=10):
    prediction = cnn_model(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as ses:
        writer = tf.train.SummaryWriter('log-tensorboard/', ses.graph)
        ses.run(tf.initialize_all_variables())
        if train:
            for epoch in range(hm_epochs):
                epoch_los = 0.
                for _ in range((int(mnist.train.num_examples / batch_size))):
                    ex, ey = mnist.train.next_batch(batch_size)
                    _, c = ses.run([optimizer, cost], feed_dict={x: ex, y: ey})
                    epoch_los += c
                print('Epoch', epoch, 'out of', hm_epochs, 'loss:', epoch_los)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('accuracy', accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))


# height x width - 64x64x3 imgs
sample_size = 784
with tf.name_scope('input'):
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')

train_cnn(x, y, True)





