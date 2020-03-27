import pickle
import gzip
import os.path
import random
import pylab as plt
import xlwt
from xlwt import Workbook

import numpy as np
from scipy.special import softmax
import sys
np.set_printoptions(threshold=sys.maxsize)

# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')

def load_data():
    f = gzip.open("mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)



def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0

training_data, validation_data, test_data = load_data()
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def relu(resultsArray):

    res = resultsArray * (resultsArray>0)
    res[res >= 1] = 0.9999
    return res

def relu_derivative(arrayOfInputs):

    return 1 * (arrayOfInputs >0)

def zeros(y2):
    y = np.zeros((y2.shape[0],10))
    for i in range(np.shape(y2)[0]):
        y[i][y2[i]] = 1
    return y

def randomWeights(x,y,range):
    return np.random.uniform(-range, range, (x, y))
    pass
def heWeights(x,y):
    return np.random.randn(x,y)*np.sqrt(2/x)
    pass
def xavierWeights(x,y):
    return np.random.randn(x,y)*np.sqrt(6/x+y)
    pass


def makeSquare(data):
    square = data.reshape((28,28))
    return square

def flatten(data):
    vector = data.reshape((data.shape[0]*data.shape[0]))
    #print('vector', vector.shape)
    return vector

def filter(x):
    return np.random.rand(x,x)

def multAndSum(x, filter):
    return np.sum(np.multiply(x,filter))

def maxPooling(x, poolSize):
    #print(int(x.shape[0]/poolSize))
    result = np.zeros( (int(x.shape[0]/poolSize),int(x.shape[0]/poolSize)))

    for i in range(result.shape[0]):
        for j in range(result.shape[0]):
            result[i,j] = np.max(x[i*poolSize:i*poolSize+poolSize, j*poolSize:j*poolSize+poolSize])

    return result

def convolution(date, filter):
    input = makeSquare(date)
    result = np.zeros((date.shape[0] - filter.shape[0] + 1, date.shape[0] - filter.shape[0] + 1))

    for i in range(result.shape[0]):
        for j in range(result.shape[0]):
            result[i,j] = multAndSum(input[i:i+filter.shape[0], j:j+filter.shape[0]],filter)

    pooled = maxPooling(result,2)
    return flatten(pooled)
def convSet(data,filter):
    result = np.array([convolution(makeSquare(x),filter) for x in data])
    return result


class NeuralNetwork:
    def __init__(self, x, y, learning_rate, function, func_deriv,filterSize,poolSize):
        self.input      = x
        self.weights1   = randomWeights(int((28-filterSize+1)/poolSize)**2,80,0.2)
        self.weights2   = randomWeights(80,10,0.2)
        self.filterSize = filterSize
        self.poolSize   = poolSize
        self.filter     = filter(filterSize)
        print('self.filter',self.filter)
        self.flat_filter = flatten(self.filter)
        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.learning_rate = learning_rate
        self.function   = function
        self.func_deriv = func_deriv


    def feedforward(self):
        #print('input',self.input.shape)
        #print('filter',self.filter.shape)
        self.convLayer = relu(convSet(self.input,self.filter))
        #print('convLayer',self.convLayer.shape)
        #print('conv', self.convLayer.shape)
        self.layer1 = self.function(np.dot(self.convLayer, self.weights1))
        #print('layer1', self.layer1.shape)
        self.output = self.function(np.dot(self.layer1, self.weights2))
        #print('output', self.output.shape)
        #print(np.average(self.output))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, self.learning_rate*((self.y - self.output) * self.func_deriv(self.output)))
        # print('dweights2',d_weights2.shape)
        #print('relu derivative ',np.average(reluDerivative(self.output)))
        error = (np.dot((self.y - self.output) * self.func_deriv(self.output), self.weights2.T) * self.func_deriv(self.layer1))
        d_weights1 = np.dot(self.convLayer.T,  self.learning_rate*error)
        # print('dweights1', d_weights1.shape)
        d_weightsFilter = np.dot(self.input.T, self.learning_rate*(np.dot(error,self.weights1.T))*relu_derivative(self.convLayer))
        # update the weights with the derivative (slope) of the loss function
        # print('dweightsfilter,', d_weightsFilter.shape)
        self.weights1 = self.weights1 + d_weights1
        self.weights2 = self.weights2 + d_weights2
        self.flat_filter = self.flat_filter + np.sum(d_weightsFilter)/np.size(d_weightsFilter)
        temp = self.flat_filter
        self.filter = temp.reshape((self.filterSize,self.filterSize))
        #print(self.filter)


if __name__ == "__main__":

    inputTraining = training_data[0]
    resultTraining = zerify(training_data[1])
    #print(resultTraining)
    inputTest = test_data[0]
    resultTest = test_data[1]

    nn = NeuralNetwork(inputTraining,resultTraining,0.015,sigmoid,sigmoid_derivative,3,2)
    nnA = NeuralNetwork(inputTraining,resultTraining,0.015,relu,relu_derivative,3,2)

    weights1 = randomWeights(784,80,0.2)
    weights2 = randomWeights(80,10,0.2)
    weights1H = heWeights(784,80)
    weights2H = heWeights(80,10)
    weights1X = xavierWeights(784,80)
    weights2X = xavierWeights(80,10)

    batch_size = 100
    wspUczenia = [0.001, 0.01, 0.1, 1]
    filterSize = [5,7,9]
    filtr = filter(3)
    w1 = randomWeights(int((28-3+1)/nn.poolSize)**2,80,0.2)
    w2 = randomWeights(80,10,0.2)

    for x in wspUczenia:
        print('wsp uczenia: ',x)
        nn.learning_rate = x
        nn.weights1 = w1
        nn.weights2 = w2
        nn.filterSize = 3
        nn.filter     = filtr
        nn.flat_filter = flatten(filtr)
        for i in range(10):
            print(i)
            for j in range(0, 5000, batch_size):
                nn.input = inputTraining[j:j + batch_size]
                nn.y = resultTraining[j:j + batch_size]
                nn.feedforward()
                nn.backprop()
            nn.input = test_data[0]
            nn.feedforward()
            number_guessed = np.argmax(nn.output, axis=1)
            print((10000 - np.count_nonzero(resultTest - number_guessed)))


    #wb.save('1xlwt example.xls')
