import numpy as np
import matplotlib.pyplot as plt
from Net import Net


def main():
    model = Net(8, 3, 8)
    x = np.eye(8)

    model.load_state_dict('./model.pth')

    output = model(x)
    plt.imshow(output, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()


