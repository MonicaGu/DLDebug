import numpy as np

import simpleflow as sf

# first, test forward. Then, test backward.
# sf.CompareSession. 
# 
import tensorflow as tf

# mnist = np.load("./mnist.npz")
mnist = np.load("/Users/gudiandian/Downloads/mnist.npz")
x_train = mnist['x_train']
x_test = mnist['x_test']
y_train = mnist['y_train']
y_test = mnist['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

batch_size = 100

def next_batch(index_in_epoch, batch_size):
	start = index_in_epoch
	end = start + batch_size
	if end > 10000:
		end = 10000 
	return {x: x_train[start:end].reshape((end - start), 784), y: y_train[start:end]}

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]


x = sf.placeholder() #tf.float32, [None, 784]
y = sf.placeholder() #tf.float32, [None, 10]


w = sf.Variable(np.random.randn(784, 10))
b = sf.Variable(np.zeros([10]))

pred = sf.add(sf.matmul(x, w), b)
pred = sf.softmax(pred)

loss = sf.mse(y, pred, framework='tf')
train_op = sf.GradientDescentOptimizer(learning_rate=0.02, framework='pytorch').minimize(loss)

index_in_epoch = 0
feed_dict_train = {x: x_train[:10000].reshape(10000, 784), y: y_train[:10000]}
feed_dict_test = {x: x_test.reshape(10000, 784), y: y_test}


with sf.DebugSession() as debug_sess:
	debug_sess.run(loss, feed_dict=feed_dict_train, frameworks=['tf', 'pytorch'])

