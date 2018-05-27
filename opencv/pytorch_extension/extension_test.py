import time


import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.jit import trace

import custom_cnn

torch.manual_seed(42)


class CustomNet(nn.Module):
    def __init__(self, num_output=6):
        super(CustomNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(20 * 4 * 8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_output)
        )
    # @profile
    def forward(self, x):
        x = self.conv_layers(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomNetCpp(nn.Module):
    def __init__(self, params):
        super(CustomNetCpp, self).__init__()
        self.params = params

    def forward(self, x):
        return custom_cnn.forward(x, self.params)

model = CustomNet()
params = []
for n, p in model.named_parameters():
    # print(n)
    params.append(p)

cpp_model = CustomNetCpp(params)

_input = torch.ones((1, 3, 41, 80)).normal_(0, 1)

model = trace(_input)(model)

result = cpp_model(_input)

print(model(_input).shape)
print(result.shape)

# print(model(_input), result)
# assert torch.equal(model(_input), result)

N_ITER = 5000

time_deltas = []
for i in range(N_ITER):
    _input = torch.ones((1, 3, 41, 80)).normal_(0, 1)
    start_time = time.time()
    # cpp_model(_input)
    model(_input)
    time_deltas.append(time.time() - start_time)

time_deltas = np.array(time_deltas)
print("Total time: {:.6f}s".format(time_deltas.sum()))
print("Mean time: {:.4f}ms".format(1000 * time_deltas.mean()))
print("Std time: {:.4f}ms".format(1000 * time_deltas.std()))
print("Median time: {:.4f}ms".format(1000 * np.median(time_deltas)))
