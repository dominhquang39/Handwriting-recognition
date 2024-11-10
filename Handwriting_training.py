import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("Handwriting_train.csv")

data = np.array(data)

m, n = data.shape

np.random.shuffle(data)


data_dev = data[0:60000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[60000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
m_train, n_train = X_train.shape

def init_parameters(layers_lists):
    list_len = len(layers_lists)
    w_b_dictionary = {}
    for i in range(1, list_len):
        w = np.random.rand(layers_lists[i], layers_lists[i - 1]) - 0.5
        b = np.random.rand(layers_lists[i], 1) - 0.5
        w_b_dictionary["w" + str(i)] = w
        w_b_dictionary["b" + str(i)] = b
    return w_b_dictionary

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_derivatite(Z):
    return Z > 0

def result(Y):
    result_Y = np.zeros((Y.size, Y.max() + 1))
    result_Y[np.arange(Y.size), Y] = 1
    result_Y = result_Y.T
    return result_Y

def forward_propagation(w_b_dictionary, X):
    number_of_z = int(len(w_b_dictionary)/2)
    z = 0
    a = X
    a_z_dictionary = {}
    for i in range(1, number_of_z):
        z = w_b_dictionary["w" + str(i)].dot(a) + w_b_dictionary["b" + str(i)]
        a = ReLU(z)
        a_z_dictionary["z" + str(i)] = z
        a_z_dictionary["a" + str(i)] = a
    z = w_b_dictionary["w" + str(number_of_z)].dot(a) + w_b_dictionary["b" + str(number_of_z)]
    a = softmax(z)
    a_z_dictionary["z" + str(number_of_z)] = z
    a_z_dictionary["a" + str(number_of_z)] = a
    return a_z_dictionary

def backward_propagation(a_z_dicitonary, w_b_dictionary, x, y):
    result_Y = result(y)

    length = int(len(w_b_dictionary)/2)

    dw_db_dictionary = {}

    dz = a_z_dicitonary["a" + str(length)] - result_Y
    dw = 1 / m * dz.dot(a_z_dicitonary["a" + str(length - 1)].T)
    db = 1/m * np.sum(dz)

    dw_db_dictionary["dw" + str(length)] = dw
    dw_db_dictionary["db" + str(length)] = db

    for i in range(length - 1, 1, -1):
        dz = w_b_dictionary["w" + str(i + 1)].T.dot(dz) * ReLU_derivatite(a_z_dicitonary["z" + str(i)]) 
        dw = 1 / m * dz.dot(a_z_dicitonary["a" + str(i - 1)].T)
        db = 1/m * np.sum(dz)
        dw_db_dictionary["dw" + str(i)] = dw
        dw_db_dictionary["db" + str(i)] = db

    dz = w_b_dictionary["w2"].T.dot(dz) * ReLU_derivatite(a_z_dicitonary["z1"]) 
    dw = 1 / m * dz.dot(x.T)
    db = 1/m * np.sum(dz)
    dw_db_dictionary["dw1"] = dw
    dw_db_dictionary["db1"] = db
    return dw_db_dictionary

def update_parameters(w_b_dictionary, dw_db_dictionary, alpha):
    length = int(len(w_b_dictionary)/2)

    for i in range(1, length + 1):
        w_b_dictionary["w" + str(i)] = w_b_dictionary["w" + str(i)] - alpha * dw_db_dictionary["dw" + str(i)]
        w_b_dictionary["b" + str(i)] = w_b_dictionary["b" + str(i)] - alpha * dw_db_dictionary["db" + str(i)]
    return w_b_dictionary

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_decent(x, y, alpha, iterations, layer_depth):

    length_layers = len(layer_depth)

    w_b_dicitonary = init_parameters(layer_depth)
    for i in range(iterations):
        a_z_dictionary = forward_propagation(w_b_dicitonary, x)
        dw_db_dictionary = backward_propagation(a_z_dictionary, w_b_dicitonary, x, y)
        w_b_dicitonary = update_parameters(w_b_dicitonary, dw_db_dictionary, alpha)
        if (i % 10 == 0):
            print("iteration: ", i)
            predicitions = get_predictions(a_z_dictionary["a" + str(length_layers - 1)])
            print(get_accuracy(predicitions, y))
    return w_b_dicitonary

w_b_dictionary = gradient_decent(X_train, Y_train, 0.1, 1000, [784, 512, 256, 128, 64, 36])

np.savetxt("data_training.csv", data_train, delimiter= ",")

np.savetxt("data_testing.csv", data_dev, delimiter= ",")

w_b_length = len(w_b_dictionary)

for i in range(1, w_b_length + 1):
    np.savetxt("w"+ str(i) + ".csv", w_b_dictionary["w" + str(i)],   delimiter= ",")
    np.savetxt("b"+ str(i) + ".csv", w_b_dictionary["b" + str(i)],   delimiter= ",")