import numpy


class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.learning_rate * parameter.grad

