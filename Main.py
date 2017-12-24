import time
from Support import load_dataset
from neuralNetwok.Model import *

mnist = load_dataset()
X_train_raw = mnist.train.images
Y_train_raw = mnist.train.labels
X_test_raw = mnist.test.images
Y_test_raw = mnist.test.labels

X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).T
print(X_train.shape)
print(X_test.shape)

Y_train = Y_train_raw.T
Y_test = Y_test_raw.T

print(Y_train.shape)
print(Y_test.shape)

model = Model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    0.00001,
    64,
    2,
    400
)
parameters = model.model()
tiempos = []
# for _ in range(4):
#     start = time.time()
#     parameters = model.model()
#     end = time.time()
#     total = end - start
#     print(total)
#     tiempos.append(total)
#
# print("Los tiempos fueron: ")
# acumulado = 0
# for i in tiempos:
#     acumulado += int(i)
#     print(i)
# print("El promedio es: ")
# promedio = acumulado / len(tiempos)
# print(promedio)
