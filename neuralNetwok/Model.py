import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from Support import *
from neuralNetwok.Layer import *


class Model:
    seed = 1

    def __init__(self, X_train, Y_train, X_test, Y_test, learning_rate, minibatch_size, num_hidden_layers, num_epochs):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_epochs = num_epochs

    def hiperparameters(self, seed):
        self.seed = seed

    def create_placeholders(self, n_x, n_y):
        X = tf.placeholder(shape=[n_x, None], dtype="float", name="X")
        Y = tf.placeholder(shape=[n_y, None], dtype="float", name="Y")
        return X, Y

    def initialize_parameters(self, name="inicializacion"):
        tf.set_random_seed(self.seed)
        parameters = {}
        layer = Layer(self.X_train.shape[0], 50)
        changer = 1
        for i in range(self.num_hidden_layers - 1):
            print(str(layer.size) + " " + str(layer.size_incoming))
            parameters["W" + str(i + 1)] = tf.get_variable \
                ("W" + str(i), [layer.size, layer.size_incoming],
                 initializer=tf.contrib.layers.xavier_initializer(seed=self.seed))
            parameters["b" + str(i + 1)] = tf.get_variable("b" + str(i), [layer.size, 1],
                                                           initializer=tf.zeros_initializer())
            layer = Layer(layer.size, layer.size + 5 * changer)
            changer *= -1
        parameters["W" + str(self.num_hidden_layers)] = tf.get_variable("W" + str(self.num_hidden_layers),
                                                                        [10, layer.size_incoming],
                                                                        initializer=tf.contrib.layers.xavier_initializer(
                                                                            seed=self.seed))
        parameters["b" + str(self.num_hidden_layers)] = tf.get_variable("b" + str(self.num_hidden_layers), [10, 1],
                                                                        initializer=tf.zeros_initializer())
        return parameters

    def forward_propagation(self, X, parameters, name="fprop"):
        A = X
        Z = None

        for i in range(self.num_hidden_layers):
            W = parameters["W" + str(i + 1)]
            b = parameters["b" + str(i + 1)]
            Z = tf.add(tf.matmul(W, A), b)
            A = tf.nn.relu(Z)

        return Z

    def compute_cost(self, Z3, Y, name="cost"):
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return cost

    def model(self, print_cost=True):
        ops.reset_default_graph()
        tf.set_random_seed(self.seed)
        internal_seed = self.seed + 1
        (n_x, m) = self.X_train.shape
        n_y = self.Y_train.shape[0]
        costos = []

        X, Y = self.create_placeholders(n_x, n_y)
        parameters = self.initialize_parameters()
        Z3 = self.forward_propagation(X, parameters)
        cost = self.compute_cost(Z3, Y)
        with tf.name_scope(name="optimizacion"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.num_epochs):
                epoch_cost = 0
                num_minibatches = int(m / self.minibatch_size)
                internal_seed += 1
                minibatches = random_mini_batches(self.X_train, self.Y_train, self.minibatch_size, internal_seed)

                for minibatch in minibatches:
                    minibatch_X, minibatch_Y = minibatch

                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches
                if print_cost == True and epoch % 100 == 0:
                    print("Costo despues del epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costos.append(epoch_cost)
            plt.interactive(False)
            plt.plot(np.squeeze(costos))
            plt.ylabel('costo')
            plt.xlabel('iteraciones (por tensor)')
            plt.title("Tasa de aprendizaje =" + str(self.learning_rate))
            # plt.show()

            parameters = sess.run(parameters)
            print("Los parametros han sido entrenados")
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Precisión entrenamiento:", accuracy.eval({X: self.X_train, Y: self.Y_train}))
            print("Precisión testedo:", accuracy.eval({X: self.X_test, Y: self.Y_test}))

            return parameters
