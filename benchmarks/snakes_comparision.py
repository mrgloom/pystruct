import numpy as np
#import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize

from pystruct.utils import make_grid_edges, edge_list_to_features
from pystruct.datasets import load_snakes
from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM, NSlackSSVM, OneSlackSSVM
from pystruct.utils import AnalysisLogger


from sklearn.utils import shuffle

def one_hot_colors(x):
    x = x / 255
    flat = np.dot(x.reshape(-1, 3),  2 **  np.arange(3))
    one_hot = label_binarize(flat, classes=[1, 2, 3, 4, 6])
    return one_hot.reshape(x.shape[0], x.shape[1], 5)


def neighborhood_feature(x):
    """Add a 3x3 neighborhood around each pixel as a feature."""
    # we could also use a four neighborhood, that would work even better
    # but one might argue then we are using domain knowledge ;)
    features = np.zeros((x.shape[0], x.shape[1], 5, 9))
    # position 3 is background.
    features[:, :, 3, :] = 1
    features[1:, 1:, :, 0] = x[:-1, :-1, :]
    features[:, 1:, :, 1] = x[:, :-1, :]
    features[:-1, 1:, :, 2] = x[1:, :-1, :]
    features[1:, :, :, 3] = x[:-1, :, :]
    features[:-1, :-1, :, 4] = x[1:, 1:, :]
    features[:-1, :, :, 5] = x[1:, :, :]
    features[1:, :-1, :, 6] = x[:-1, 1:, :]
    features[:, :-1, :, 7] = x[:, 1:, :]
    features[:, :, :, 8] = x[:, :, :]
    return features.reshape(x.shape[0] * x.shape[1], -1)


def prepare_data(X):
    X_directions = []
    X_edge_features = []
    for x in X:
        # get edges in grid
        right, down = make_grid_edges(x, return_lists=True)
        edges = np.vstack([right, down])
        # use 3x3 patch around each point
        features = neighborhood_feature(x)
        # simple edge feature that encodes just if an edge is horizontal or
        # vertical
        edge_features_directions = edge_list_to_features([right, down])
        # edge feature that contains features from the nodes that the edge connects
        edge_features = np.zeros((edges.shape[0], features.shape[1], 4))
        edge_features[:len(right), :, 0] = features[right[:, 0]]
        edge_features[:len(right), :, 1] = features[right[:, 1]]
        edge_features[len(right):, :, 0] = features[down[:, 0]]
        edge_features[len(right):, :, 1] = features[down[:, 1]]
        edge_features = edge_features.reshape(edges.shape[0], -1)
        X_directions.append((features, edges, edge_features_directions))
        X_edge_features.append((features, edges, edge_features))
    return X_directions, X_edge_features


snakes = load_snakes()
X_train, Y_train = snakes['X_train'], snakes['Y_train']

X_train = [one_hot_colors(x) for x in X_train]
Y_train_flat = [y_.ravel() for y_ in Y_train]

X_train_directions, X_train_edge_features = prepare_data(X_train)


inference = ('ogm', {'alg': 'fm'})
model = EdgeFeatureGraphCRF(inference_method=inference)

bcfw = FrankWolfeSSVM(model=model, C=.1, max_iter=100, tol=0.1, verbose=3, check_dual_every=3, averaging='linear')
pegasos = SubgradientSSVM(model=model, C=.1, max_iter=100, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=len(X_train) * .1)
pystructsgd = SubgradientSSVM(model=model, C=.1, max_iter=100, verbose=3)
nslack = NSlackSSVM(model, C=.1, tol=.1, verbose=3)
nslack_every = NSlackSSVM(model, C=.1, tol=.1, verbose=3, batch_size=1)
oneslack = OneSlackSSVM(model, C=.1, tol=.1, verbose=3)
oneslack_cache = OneSlackSSVM(model, C=.1, tol=.1, inference_cache=50, verbose=3)

svms = [bcfw, pystructsgd, oneslack, oneslack_cache, pegasos, nslack, nslack_every]
names = ['bcfw', 'pystructsgd_momentum', "oneslack", "oneslack_cache", "pegasos", "nslack", "nslack_every"]

X_train_edge_features, Y_train_flat = shuffle(X_train_edge_features, Y_train_flat)

for name, svm in zip(names, svms):
    logger = AnalysisLogger("snakes_fm_ana_logger_" + name + ".pickle", log_every=10)
    svm.logger = logger
    svm.fit(X_train_edge_features, Y_train_flat)
