import numpy as np

from train.train import loadNetwork


network, pred_fn = loadNetwork(cnn=False)

weights_npy = 'mlp_model.npz'

with np.load(weights_npy) as f:
    print("Exporting pre-trained params")
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]


W1, b1, W2, b2, W3, b3, W4, b4 = param_values

X = np.ones((1, 2880))
X = np.random.random((1, 2880))

def relu(x):
    y = x.copy()
    y[y<0] = 0
    return y

def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)
    Z4 = np.dot(A3, W4) + b4
    A4 = relu(Z4)
    return A4

print(pred_fn(X).ravel()[0])
print(forward(X).ravel()[0])
