import numpy as np
import torch as th
from torch import nn

h, w, c = 32, 32, 3

image = np.random.random((1, c, h, w)).astype(np.float32)

def zeroPad(x, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    """

    padding = tuple([pad] * 2)
    x_pad = np.pad(x, ((0,0), (0,0), padding, padding), 'constant')

    return x_pad

def relu(x):
    """
    Rectify activation function: f(x) = max(0, x)
    :param x: (numpy array)
    :return: (numpy array)
    """
    y = x.copy()
    y[y < 0] = 0
    return y

def conv2d(x, in_channels, out_channels, kernel_size=3, stride=1, padding=0, kernel=None, bias=None):
    m, c_prev, h_prev, w_prev = x.shape

    if kernel is None:
        kernel = np.ones((out_channels, in_channels, kernel_size, kernel_size))
        bias = np.ones((out_channels, ))

    out_w = int((w_prev - kernel_size + 2 * padding) / stride + 1)
    out_h = int((h_prev - kernel_size + 2 * padding) / stride + 1)

    # Initialize the output volume with zeros
    z = np.zeros((m, out_channels, out_h, out_w))

    # Pad input
    x_pad = zeroPad(x, padding)

    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(out_channels):
                    # Compute slice
                    vert_start = h * stride
                    vert_end = vert_start + kernel_size
                    horiz_start = w * stride
                    horiz_end = horiz_start + kernel_size
                    # x_slice = x_prev_pad[i, :, vert_start:vert_end, horiz_start:horiz_end]
                    x_slice = x_pad[i, :, vert_start:vert_end, horiz_start:horiz_end]
                    # Convolve the (3D) slice with the correct kernel and bias, to get back one output neuron.
                    # Element-wise product between x_slice and kernel
                    s = kernel[c, :, :, :] * x_slice
                    # Sum over all entries of the volume s and add bias
                    z[i, c, h, w] = s.sum() + bias[c]
    return z

def pool2d(x, kernel_size=3, stride=1, padding=0, mode="max"):
    m, c_prev, h_prev, w_prev = x.shape

    out_w = int((w_prev - kernel_size + 2 * padding) / stride + 1)
    out_h = int((h_prev - kernel_size + 2 * padding) / stride + 1)
    out_channels = c_prev

    # Initialize the output volume with zeros
    z = np.zeros((m, out_channels, out_h, out_w))

    # Pad input
    x_pad = zeroPad(x, padding)

    for i in range(m):
        for h in range(out_h):
            for w in range(out_w):
                for c in range(out_channels):
                    # Compute slice
                    vert_start = h * stride
                    vert_end = vert_start + kernel_size
                    horiz_start = w * stride
                    horiz_end = horiz_start + kernel_size

                    x_slice = x_pad[i, c, vert_start:vert_end, horiz_start:horiz_end]
                    if mode == "max":
                        z[i, c, h, w] = x_slice.max()
                    elif mode == "average":
                        z[i, c, h, w] = x_slice.mean()
    return z

def numpyNet(x, params):
    z = conv2d(x, 3, 20, kernel_size=7, stride=2, padding=1, kernel=params[0], bias=params[1])
    z = relu(z)
    z = pool2d(z, kernel_size=3, stride=2)

    z = conv2d(z, 20, 20, kernel_size=3, stride=1, padding=1, kernel=params[2], bias=params[3])
    z = relu(z)
    z = pool2d(z, kernel_size=3, stride=2)
    return z

class CustomNet(nn.Module):
    def __init__(self, num_output=6):
        super(CustomNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(20*4*8, 32),
        #     # nn.Linear(20*9*18, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, num_output)
        # )

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


model = CustomNet()
params = []
for n, p in model.named_parameters():
    print(n)
    params.append(p.detach().numpy())

a = model(th.from_numpy(image)).detach().numpy()
b = numpyNet(image, params)
print(a.shape, b.shape)
# print((a - b)[0, 0, 0])
