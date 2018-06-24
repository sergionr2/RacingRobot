from __future__ import print_function, division, absolute_import

import json

import cv2
import numpy as np
import torch as th
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from constants import MAX_WIDTH, MAX_HEIGHT, ROI, INPUT_HEIGHT, INPUT_WIDTH, SPLIT_SEED, N_CHANNELS
from .models import ConvolutionalNetwork, CustomNet, MlpNetwork


def adjustLearningRate(optimizer, epoch, n_epochs, lr_init, batch,
                       n_batch, method='cosine', decay_rate=0.7):
    """
    :param optimizer: (PyTorch Optimizer object)
    :param epoch: (int)
    :param n_epochs: (int)
    :param lr_init: (float)
    :param batch: (int)
    :param n_batch: (int)
    :param method: (str) one of ("cosine", "multistep")
    :param decay_rate: (float)
    """
    if method == 'cosine':
        T_total = n_epochs * n_batch
        T_cur = (epoch % n_epochs) * n_batch + batch
        lr = 0.5 * lr_init * (1 + np.cos(np.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = lr_init
        if epoch >= n_epochs * 0.75:
            lr *= decay_rate ** 2
        elif epoch >= n_epochs * 0.5:
            lr *= decay_rate
    # else:
    #     # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    #     lr = lr_init * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def predict(model, image):
    """
    :param model: (PyTorch Model)
    :param image: (numpy tensor)
    :return: (numpy array, numpy array)
    """
    im = preprocessImage(image, INPUT_WIDTH, INPUT_HEIGHT)
    # Re-order channels for pytorch
    im = im.transpose((2, 0, 1)).astype(np.float32)
    with th.no_grad():
        predictions = model(th.from_numpy(im[None]))[0].data.numpy()
    x, y = transformPrediction(predictions)
    return x, y


def loadNetwork(weights, num_output=6, model_type="cnn"):
    """
    :param weights: (str)
    :param num_output: (int)
    :param model_type: (str)
    :return: (PyTorch Model)
    """
    if model_type == "cnn":
        model = ConvolutionalNetwork(num_output=num_output)
    elif model_type == "custom":
        model = CustomNet(num_output=num_output)
    elif model_type == "mlp":
        model = MlpNetwork(INPUT_WIDTH * INPUT_HEIGHT * N_CHANNELS, num_output=num_output)

    model.load_state_dict(th.load(weights))
    model.eval()
    return model


def preprocessImage(image, width, height):
    """
    Preprocessing script to convert image into neural net input array
    :param image: (cv2 image)
    :param width: (int)
    :param height: (int)
    :return: (numpy tensor)
    """
    # Crop the image
    r = ROI
    image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # The resizing is a bottleneck in the computation
    x = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Normalize
    x = x / 255.
    x -= 0.5
    x *= 2
    return x


def transformPrediction(y):
    """
    Transform the model output back
    to original image space (pixel position)
    :param y: (numpy array)
    :return: (numpy array, numpy array)
    """
    margin_left, margin_top, _, _ = ROI
    points = y.flatten()
    x = points[::2]
    y = points[1::2]
    y = (y * MAX_HEIGHT) + margin_top
    x = (x * MAX_WIDTH) + margin_left
    return x, y


def loadLabels(folders):
    if not isinstance(folders, list):
        folders = [folders]

    labels = {}
    images = []
    for folder in folders:
        if not folder.endswith('/'):
            folder += '/'
        tmp_labels = json.load(open(folder + 'labels.json'))
        tmp_images = list(tmp_labels.keys())
        tmp_images.sort(key=lambda name: int(name.split('.jpg')[0]))
        tmp_labels = {"{}/{}".format(folder, name): tmp_labels[name] for name in tmp_images}
        labels.update(tmp_labels)
        tmp_images = ["{}/{}".format(folder, name) for name in tmp_images]
        images += tmp_images

    # Split the data into three subsets
    train_keys, tmp_keys = train_test_split(images, test_size=0.4, random_state=SPLIT_SEED)
    val_keys, test_keys = train_test_split(tmp_keys, test_size=0.5, random_state=SPLIT_SEED)

    train_labels = {key: labels[key] for key in train_keys}
    val_labels = {key: labels[key] for key in val_keys}
    test_labels = {key: labels[key] for key in test_keys}

    print("{} labels".format(len(labels)))
    return train_labels, val_labels, test_labels, labels


class JsonDataset(Dataset):
    """
    :param labels: (dict)
    :param preprocess: (bool)
    :param random_flip: (float) probability of flipping the image
    """

    def __init__(self, labels, preprocess=False, random_flip=0.0):
        self.keys = list(labels.keys())
        self.labels = labels.copy()
        self.preprocess = preprocess
        self.random_flip = random_flip

    def __getitem__(self, index):
        """
        :param index: (int)
        :return: (th.Tensor, th.Tensor)
        """
        image = self.keys[index]
        margin_left, margin_top = 0, 0
        im = cv2.imread(image)

        # Crop the image and normalize it
        if self.preprocess:
            margin_left, margin_top, _, _ = ROI
            im = preprocessImage(im, INPUT_WIDTH, INPUT_HEIGHT)

        # Normalize labels
        labels = np.array(self.labels[image]).astype(np.float32)
        assert len(labels) == 3, "Error with label of image {}".format(image)
        labels[:, 0] = (labels[:, 0] - margin_left) / MAX_WIDTH
        labels[:, 1] = (labels[:, 1] - margin_top) / MAX_HEIGHT

        # Randomly flip the image to augment the dataset
        if np.random.random() < self.random_flip:
            im = cv2.flip(im, 1)
            labels[:, 0] = 1 - labels[:, 0]

        # 3 points -> 6 values to predict
        y = labels.flatten().astype(np.float32)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1)).astype(np.float32)
        return th.from_numpy(im), th.from_numpy(y)

    def __len__(self):
        return len(self.keys)


def computeLossWithDataLoader(model, labels, batchsize, shuffle=False):
    """
    :param model: (Pytorch Model)
    :param labels: (dict)
    :param batchsize: (int)
    :param shuffle: (bool)
    :return: (float)
    """
    dataloader = th.utils.data.DataLoader(JsonDataset(labels, preprocess=True),
                                          batch_size=batchsize, shuffle=shuffle)

    loss_fn = th.nn.MSELoss(size_average=True)
    total_loss = 0
    with th.no_grad():
        for inputs, targets in dataloader:
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
    return total_loss


def computeMSE(model, train_labels, val_labels, test_labels, batchsize=32):
    """
    Compute Mean Square Error
    and print its value for the different sets
    :param model: (Pytorch Model)
    :param train_labels: (dict)
    :param val_labels: (dict)
    :param test_labels: (dict)
    :param batchsize: (int)
    """
    model.eval()
    error_train = computeLossWithDataLoader(model, train_labels, batchsize)
    error_val = computeLossWithDataLoader(model, val_labels, batchsize)
    error_test = computeLossWithDataLoader(model, test_labels, batchsize)

    print('Train error={:.6f}'.format(error_train))
    print('Val error={:.6f}'.format(error_val))
    print('Test error={:.6f}'.format(error_test))
    print('Total error={:.6f}'.format((error_train + error_val + error_test) / 3))
