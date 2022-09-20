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

    def train(self, x, y, epochs=100):
        a, b = x.shape 
        for e in range(epochs):
            for i in range(b):
                
                xtemp = [x[0][i], x[1][i]]
                prediction = self.activation(xtemp)
                #print(f"y = {y[i]}, prediction = {prediction}")
                error = y[i] - prediction
                self.weights[0] += self.lr*x[0][i]*error
                self.weights[1] += self.lr*x[1][i]*error
                self.b += self.lr * error

            self.totalW1.append(self.weights[0])
            self.totalW2.append(self.weights[1])
            self.totalB.append(self.b)
            self.totalError.append(error)
        
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


samples = 100
learningRate = 0.1

x = np.zeros((2, samples))
y = np.zeros(samples)

for i in range(samples):
    h = 1.2 + (2.4 - 1.2) * np.random.rand()
    p = 40 + (100 - 40) * np.random.rand()
    imc = p / (h**2)

    if imc > 25:
        ytemp = 1
    else:
        ytemp = 0
    y[i] = ytemp
    x[0][i] = p
    x[1][i] = h

for i in range(samples):
    x[0][i] = (x[0][i] - x[0].min()) / (x[0].max() - x[0].min())
    x[1][i] = (x[1][i] - x[1].min()) / (x[1].max() - x[1].min())

model = Perceptron(2, learningRate)
model.train(x, y)

for i in range(samples):
    if y[i] == 0:
        plt.plot(x[0][i], x[1][i], 'ob')
    else:
        plt.plot(x[0][i], x[1][i], 'or')

plt.title('Perceptrón')
plt.grid('on')
plt.xlim([0, 1.25])
plt.ylim([0, 1.25])
plt.xlabel(r'Peso (kg)')
plt.ylabel(r'Altura (m)')

model.plotModel()
plt.show()