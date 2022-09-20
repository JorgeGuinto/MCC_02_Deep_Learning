import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    totalW1 = []
    totalW2 = []
    totalB = []
    totalError = []

    def __init__(self, n, learningRate):
        self.weights = np.random.rand(n)
        self.b = np.random.rand()
        self.lr = learningRate


    def activation(self, x):
        z = np.dot(self.weights, x)
        if (z.sum() + self.b) > 0:
            return 1
        else:
            return 0

    def train(self, x, y, epochs=20):
        for e in range(epochs):
            for i in range(len(x)):
                
                prediction = self.activation(x[i])
                error = y[i] - prediction

                self.totalW1.append(self.weights[0])
                self.totalW2.append(self.weights[1])
                self.totalB.append(self.b)
                self.totalError.append(error)
                
                self.weights[0] += self.lr*x[i][0]*error
                self.weights[1] += self.lr*x[i][1]*error
                self.b += self.lr * error
        plt.title("Evolucion")
        plt.plot(self.totalW1, label = "w1")
        plt.plot(self.totalW2, label = "w2")
        plt.plot(self.totalB, label = "b")
        plt.plot(self.totalError, label = "Error")
        plt.legend()
        plt.show()

    def plotModel(self):
        w1, w2, b = self.weights[0], self.weights[1], self.b
        plt.plot([-2,2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*2-b)], '--k')





x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#AND
#y = np.array([0, 0, 0, 1])
#OR
#y = np.array([0, 1, 1, 1])
#XOR
y = np.array([0, 1, 1, 0])
model = Perceptron(2, 0.1)
model.train(x, y)

a, p = x.shape
for i in range(a):
    if y[i] == 0:
        plt.plot(x[i, 0], x[i, 1], 'or')
    else:
        plt.plot(x[i, 0], x[i, 1], 'ob')

plt.title('Perceptrón')
plt.grid('on')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

model.plotModel()
plt.show()


