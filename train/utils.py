from __future__ import print_function, division, absolute_import

import pickle as pkl

import cv2
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split

from constants import INPUT_DIM, HEIGHT, WIDTH
from .models import MlpNetwork


def adjustLearningRate(optimizer, epoch, n_epochs, lr_init, batch,
                         n_batch, method='cosine'):
    """
    :param optimizer: (PyTorch Optimizer object)
    :param epoch: (int)
    :param n_epochs: (int)
    :param lr_init: (float)
    :param batch: (int)
    :param n_batch: (int)
    :param method: (str)
    """
    if method == 'cosine':
        T_total = n_epochs * n_batch
        T_cur = (epoch % n_epochs) * n_batch + batch
        lr = 0.5 * lr_init * (1 + np.cos(np.pi * T_cur / T_total))
    elif method == 'multistep':
        lr, decay_rate = lr_init, 0.7
        if epoch >= n_epochs * 0.75:
            lr *= decay_rate ** 2
        elif epoch >= n_epochs * 0.5:
            lr *= decay_rate
    # else:
    #     # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    #     lr = lr_init * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def loadPytorchNetwork(model_name="mlp_model_tmp", n_hidden=None, model_type="mlp"):
    """
    Load a saved pytorch model
    :param model_name: (str)
    :param n_hidden: ([int])
    :return: (pytorch model)
    """
    if '.pth' in model_name:
        model_name = model_name.split('.pth')[0]
    if model_type == "mlp":
        model = MlpNetwork(INPUT_DIM, n_hidden=n_hidden)
    else:
        model = ConvolutionalNetwork()
    model.load_state_dict(th.load(model_name + '.pth'))
    model.eval()
    return model


def saveToNpz(model, output_name="mlp_model_tmp"):
    """
    :param model: (PyTorch Model)
    :param output_name: (str)
    """
    np.savez(output_name, *[p.data.numpy().T for _, p in model.named_parameters()])


def loadDataset(split_seed=42, folder='', split=True, augmented=True, num_stack=1):
    """
    Load the training images and preprocess them
    :param split_seed: (int) split_seed for pseudo-random generator
    :param folder: (str) input folder
    :param split: (bool) Whether to split the dataset into 3 subsets (train, validation, test)
    :param augmented: (bool) Whether to use data augmentation
    :param num_stack: (int)
    :return:
    """

    # Load the dataset info file (pickle object)
    with open('{}/infos.pkl'.format(folder), 'rb') as f:
        try:
            images_dict = pkl.load(f)['images']
        except UnicodeDecodeError:
            # (python 2 -> python 3)
            images_dict = pkl.load(f, encoding='latin1')['images']

    # Sort names
    images = list(images_dict.keys())
    images.sort(key=lambda name: int(images_dict[name]['output_name']))
    images_path = []

    # Load one image to retrieve original shape
    tmp_im = cv2.imread('{}/{}.jpg'.format(folder, images_dict[images[0]]['output_name']))
    height, width, _ = tmp_im.shape
    n_images = len(images)
    # If we use data augmentation we double the size of training data
    if augmented:
        images_path_augmented = []
        n_images *= 2

    X = np.zeros((n_images, INPUT_DIM), dtype=np.float32)
    y = np.zeros((n_images,), dtype=np.float32)

    print("original_shape=({},{})".format(width, height))
    print("resized_shape=({},{})".format(WIDTH, HEIGHT))

    for idx, name in enumerate(images):
        x_center, y_center = images_dict[name]['label']
        # Normalize output
        y[idx] = x_center / width

        path = images_dict[name]['output_name']
        image_path = '{}/{}.jpg'.format(folder, path)
        im = cv2.imread(image_path)
        # Resize and normalize input
        X[idx, :] = preprocessImage(im, WIDTH, HEIGHT)
        images_path.append(path + '.jpg')
        # Flip the image+label to have more training data
        if augmented:
            horizontal_flip = cv2.flip(im, 1)
            X[len(images) + idx, :] = preprocessImage(horizontal_flip, WIDTH, HEIGHT)
            y[len(images) + idx] = (width - x_center) / width
            images_path_augmented.append(path + '.jpg')

    # By convention, augmented data are at the end
    if augmented:
        images_path += images_path_augmented

    if num_stack > 1:
        X_stack = np.zeros((n_images, num_stack * INPUT_DIM), dtype=np.float32)
        for i in range(n_images):
            X_stack[i, :INPUT_DIM] = X[i]

        # TODO: skip transition between flip and normal frames
        num_skipped = 0
        for i in range(n_images - 1, num_stack, -1):
            input_image_idx, input_region = map(int, images[i % len(images)].split('.jpg_r'))
            for k in range(num_stack):
                prev_frame_idx = input_image_idx - k - 1
                j = i - 1
                ok = False
                # Find the same region in the next frame
                while j >= 0:
                    image_idx, region = map(int, images[j % len(images)].split('.jpg_r'))

                    if image_idx == prev_frame_idx and region == input_region:
                        ok = True
                        break
                    if image_idx < prev_frame_idx:
                        break
                    j -= 1
                if not ok:
                    num_skipped += 1
                    # print("Skipping frame stacking for {}".format(images[i % len(images)]))
                    break
                X_stack[i, k*INPUT_DIM:(k + 1) * INPUT_DIM] = X[j]

        print("{:.2f}% skipped".format(num_skipped / n_images))
        X = X_stack

    print("Input tensor shape: ", X.shape)

    if not split:
        return X, y, images_path

    # Split the data into three subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=split_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=split_seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocessImage(image, width, height):
    """
    Preprocessing script to convert image into neural net input array
    :param image: (cv2 image)
    :param width: (int)
    :param height: (int)
    :return: (numpy array)
    """
    # The resizing is a bottleneck in the computation
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    x = image.flatten()
    # Normalize
    x = x / 255.
    x -= 0.5
    x *= 2
    return x


def loadWeights(weights_npy='mlp_model.npz'):
    """
    Load and return weights of a trained model
    :param weights_npy: (str) path to the numpy file
    :return: (dict, dict)
    """
    # Load pretrained network
    W, b = {}, {}
    with np.load(weights_npy) as f:
        n_layers = len(f.files) // 2
        for i in range(len(f.files)):
            # print(f['arr_%d' % i].shape)
            if i % 2 == 1:
                b[i // 2] = f['arr_%d' % i].astype(np.float32)
            else:
                W[i // 2] = f['arr_%d' % i].astype(np.float32)
    return W, b


def loadVanillaNet(weights_npy='mlp_model.npz'):
    """
    Load a trained network and
    return the forward function in pure numpy
    :param weights_npy: (str) path to the numpy file
    :return: (function) the neural net forward function
    """
    W, b = loadWeights(weights_npy)
    n_layers = len(W)

    def relu(x):
        """
        Rectify activation function: f(x) = max(0, x)
        :param x: (numpy array)
        :return: (numpy array)
        """
        y = x.copy()
        y[y < 0] = 0
        return y

    def forward(X):
        """
        Forward pass of a fully-connected neural net
        with rectifier activation function
        :param X: (numpy tensor)
        :return: (numpy array)
        """
        a = X
        for i in range(n_layers):
            z = np.dot(a, W[i]) + b[i]
            a = relu(z)
        return a

    return forward


def computeMSE(y_test, y_true, indices):
    """
    Compute Mean Square Error
    and print its value for the different sets
    :param y_test: (numpy 1D array)
    :param y_true: (numpy 1D array)
    :param indices: [[int]] Indices of the different subsets
    """
    idx_train, idx_val, idx_test = indices
    # MSE Loss
    error = np.square(y_test - y_true)

    print('Train error={:.6f}'.format(np.mean(error[idx_train])))
    print('Val error={:.6f}'.format(np.mean(error[idx_val])))
    print('Test error={:.6f}'.format(np.mean(error[idx_test])))
    print('Total error={:.6f}'.format(np.mean(error)))
