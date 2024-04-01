import numpy as np
import nn.module as nn
from loss.MSE import MeanSquaredError
from optim.Adam import Adam
from optim.SGD import SGD
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x

    def backward(self, grad_output):
        grad_output = self.sigmoid2.backward(grad_output)
        grad_output = self.linear2.backward(grad_output)
        grad_output = self.sigmoid1.backward(grad_output)
        grad_output = self.linear1.backward(grad_output)


def main():
    model = Net(8, 3, 8)
    x = np.eye(8)
    t = np.eye(8)
    optimizer = SGD(model.parameters(), 10000000)
    criterion = MeanSquaredError()
    loss_list = []
    model.load_state_dict('./model.pth')
    for i in range(10000000):
        y = model(x)
        loss = criterion(y, t)
        loss_list.append(loss)
        model.backward(criterion.backward(y, t))
        optimizer.step()
        if i % 10000 == 0:
            print(f'epoch {i}, loss {loss}')
            model.save(model.state_dict(), './model.pth')

    plt.plot(loss_list)
    plt.title('My SGD, learning rate = 0.01')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    output = model(x)
    plt.imshow(output, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
