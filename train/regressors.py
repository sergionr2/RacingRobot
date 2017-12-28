from __future__ import print_function, division, absolute_import

import argparse
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
# from xgboost import XGBRegressor

from constants import SPLIT_SEED
from .train import loadDataset
from .utils import computeMSE


parser = argparse.ArgumentParser(description='Train a scikit-learn regressor')
parser.add_argument('-f', '--folder', help='Training folder', type=str, required=True)
parser.add_argument('-m', '--model', help='Model type', default="random_forest", type=str)
parser.add_argument('--no-data-augmentation', action='store_true', default=False, help='Disables data augmentation')

args = parser.parse_args()

augmented = not args.no_data_augmentation
# Load dataset
X, y_true, images = loadDataset(folder=args.folder, split=False, augmented=augmented)

indices = np.arange(len(y_true))
idx_train, idx_test = train_test_split(indices, test_size=0.4, random_state=SPLIT_SEED)
idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=SPLIT_SEED)

print(args.model)

if args.model == "svm":
    X = PCA(n_components=400).fit_transform(X)
    model = SVR(kernel='rbf', epsilon=0.0001, gamma=0.002, C=1.0, max_iter=12000, verbose=1)
elif args.model == "random_forest":
    model = RandomForestRegressor(n_estimators=40, max_depth=15, random_state=0, verbose=1, n_jobs=-1)
# elif args.model == "xgboost":
#     model = XGBRegressor(max_depth=15, learning_rate=0.1, n_estimators=80, silent=True, nthread=8)
elif args.model == "knn":
    X = PCA(n_components=150).fit_transform(X)
    model = KNeighborsRegressor(n_neighbors=3, weights="distance", algorithm="auto",
                                leaf_size=30, p=2, metric='minkowski', n_jobs=-1)
else:
    raise ValueError("Unknown model type: {}".format(args.model))


model.fit(X[idx_train], y_true[idx_train])

start_time = time.time()
y_test = model.predict(X)
total_time = time.time() - start_time
print("\nTime to predict: {:.2f}s | {:.5f} ms/image".format(total_time, 1000 * total_time / len(y_true)))

computeMSE(y_test, y_true, [idx_train, idx_val, idx_test])
