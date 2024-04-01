import numpy as np
import pickle


class Module:
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    @staticmethod
    def save(state_dict, path):
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    def load_state_dict(self, path):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        parameters = self.parameters()
        for i, p in enumerate(state_dict):
            for key, value in p.items():
                if key == 'data':
                    setattr(parameters[i], key, value)

    def state_dict(self):
        state_list = []
        for p in self.parameters():
            state_dict = {}
            for key, value in p.__dict__.items():
                state_dict[key] = value
            state_list.append(state_dict)
        return state_list

    def parameters(self):
        if isinstance(self, Module):
            parameters = []
            for m in self.__dict__.values():
                if isinstance(m, Module):
                    for p in m.__dict__.values():
                        if isinstance(p, Parameter):
                            parameters.append(p)
                elif isinstance(m, Parameter):
                    parameters.append(m)

            if parameters:
                return parameters
            else:
                raise AttributeError('No parameters found in the model')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


class Linear(Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Linear, self).__init__()
        self.weights = Parameter(np.random.randn(input_size, output_size))
        self.bias = Parameter(np.random.randn(output_size) if bias else 0)
        self.input = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights.data) + self.bias.data

    def backward(self, grad_output):
        if self.input.ndim == 1:
            self.input = self.input.reshape(1, -1)
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)
        self.weights.grad = np.einsum('ij,ik->jk', self.input, grad_output)
        self.bias.grad = np.sum(grad_output, axis=0) if self.bias is not None else 0
        return np.einsum('ij,kj->ik', grad_output, self.weights.data)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
