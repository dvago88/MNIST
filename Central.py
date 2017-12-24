# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.python.framework import ops
# from Support import *
#
# seed = 1
#
# mnist = load_dataset()
# X_train_raw = mnist.train.images
# Y_train_raw = mnist.train.labels
# X_test_raw = mnist.test.images
# Y_test_raw = mnist.test.labels
#
# X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T
# X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).T
# print(X_train.shape)
# print(X_test.shape)
#
# Y_train = Y_train_raw.T
# Y_test = Y_test_raw.T
#
# print(Y_train.shape)
# print(Y_test.shape)
#
#
# def hiperparameters():
#     seed = 1
#
#
# def create_placeholders(n_x, n_y):
#     X = tf.placeholder(shape=[n_x, None], dtype="float", name="X")
#     Y = tf.placeholder(shape=[n_y, None], dtype="float", name="Y")
#     return X, Y
#
#
# def initialize_parameters(name="inicializacion"):
#     tf.set_random_seed(seed)
#     with tf.name_scope(name):
#         W1 = tf.get_variable("W1", [25, 784], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
#         b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
#         W2 = tf.get_variable("W2", [20, 25], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
#         b2 = tf.get_variable("b2", [20, 1], initializer=tf.zeros_initializer())
#         W3 = tf.get_variable("W3", [10, 20], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
#         b3 = tf.get_variable("b3", [10, 1], initializer=tf.zeros_initializer())
#
#         parameters = {
#             "W1": W1,
#             "b1": b1,
#             "W2": W2,
#             "b2": b2,
#             "W3": W3,
#             "b3": b3
#         }
#         return parameters
#
#
# def forward_propagation(X, parameters, name="fprop"):
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]
#     W2 = parameters["W2"]
#     b2 = parameters["b2"]
#     W3 = parameters["W3"]
#     b3 = parameters["b3"]
#
#     with tf.name_scope(name):
#         Z1 = tf.add(tf.matmul(W1, X), b1)
#         A1 = tf.nn.relu(Z1)
#         Z2 = tf.add(tf.matmul(W2, A1), b2)
#         A2 = tf.nn.relu(Z2)
#         Z3 = tf.add(tf.matmul(W3, A2), b3)
#     return Z3
#
#
# def compute_cost(Z3, Y, name="cost"):
#     with tf.name_scope(name):
#         logits = tf.transpose(Z3)
#         labels = tf.transpose(Y)
#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
#     return cost
#
#
# def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
#           num_epochs=1000, minibatch_size=64, print_cost=True):
#     ops.reset_default_graph()
#     tf.set_random_seed(seed)
#     internal_seed = seed + 1
#     (n_x, m) = X_train.shape
#     n_y = Y_train.shape[0]
#     costos = []
#
#     X, Y = create_placeholders(n_x, n_y)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     with tf.name_scope(name="optimizacion"):
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         merge_summary = tf.summary.merge_all()
#         file_writer = tf.summary.FileWriter('/logs1', sess.graph)
#         sess.run(init)
#         for epoch in range(num_epochs):
#             epoch_cost = 0
#             num_minibatches = int(m / minibatch_size)
#             internal_seed += 1
#             minibatches = random_mini_batches(X_train, Y_train, minibatch_size, internal_seed)
#
#             for minibatch in minibatches:
#                 minibatch_X, minibatch_Y = minibatch
#
#                 # if epoch % 5 == 0:
#                 #     s = sess.run(merge_summary, feed_dict={X: minibatch_X, Y: minibatch_Y})
#                 #     file_writer.add_summary(s, epoch)
#
#                 _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
#                 epoch_cost += minibatch_cost / num_minibatches
#             if print_cost == True and epoch % 100 == 0:
#                 print("Costo despues del epoch %i: %f" % (epoch, epoch_cost))
#             if print_cost == True and epoch % 5 == 0:
#                 costos.append(epoch_cost)
#         plt.interactive(False)
#         plt.plot(np.squeeze(costos))
#         plt.ylabel('costo')
#         plt.xlabel('iteraciones (por tensor)')
#         plt.title("Tasa de aprendizaje =" + str(learning_rate))
#         plt.show()
#
#         parameters = sess.run(parameters)
#         print("Los parametros han sido entrenados")
#         correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#         print("Precisión entrenamiento:", accuracy.eval({X: X_train, Y: Y_train}))
#         print("Precisión testedo:", accuracy.eval({X: X_test, Y: Y_test}))
#
#         return parameters
#
# parameters = model(X_train, Y_train, X_test, Y_test)
