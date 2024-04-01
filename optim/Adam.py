import numpy


class Adam:
    def __init__(self, parameters, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [numpy.zeros_like(parameter.data) for parameter in self.parameters]
        self.v = [numpy.zeros_like(parameter.data) for parameter in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for parameter, m, v in zip(self.parameters, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * parameter.grad
            v[:] = self.beta2 * v + (1 - self.beta2) * parameter.grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            parameter.data -= self.learning_rate * m_hat / (numpy.sqrt(v_hat) + self.epsilon)