import numpy as np

from opencv.train.train import loadNetwork

network, pred_fn = loadNetwork()

weights_npy = 'mlp_model.npz'

W, b = {}, {}
with np.load(weights_npy) as f:
    print("Exporting pre-trained params")
    n_layers = len(f.files) // 2
    for i in range(len(f.files)):
        if i % 2 == 1:
            b[i // 2] = f['arr_%d' % i]
        else:
            W[i // 2] = f['arr_%d' % i]


def relu(x):
    y = x.copy()
    y[y < 0] = 0
    return y


def forward(X):
    a = X
    for i in range(n_layers):
        z = np.dot(a, W[i]) + b[i]
        a = relu(z)
    return a


# X = np.ones((1, 2880))
X = np.random.random((1, 2880))

print(pred_fn(X).ravel()[0])
print(forward(X).ravel()[0])
