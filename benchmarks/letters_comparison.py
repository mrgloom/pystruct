import numpy as np
#import matplotlib.pyplot as plt

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM, NSlackSSVM, OneSlackSSVM
from pystruct.utils import SaveLogger


from sklearn.utils import shuffle

abc = "abcdefghijklmnopqrstuvwxyz"

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds != 1], X[folds == 1]
y_train, y_test = y[folds != 1], y[folds == 1]


# Train linear chain CRF
model = ChainCRF(inference_method=('ogm', {'alg': 'dyn'}))

bcfw = FrankWolfeSSVM(model=model, C=.1, max_iter=100, tol=0.1, verbose=3, check_dual_every=3, averaging='linear')
pegasos = SubgradientSSVM(model=model, C=.1, max_iter=100, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=len(X_train) * .1)
pystructsgd = SubgradientSSVM(model=model, C=.1, max_iter=100, verbose=3)
nslack = NSlackSSVM(model, C=.1, tol=.1, verbose=3)
nslack_every = NSlackSSVM(model, C=.1, tol=.1, verbose=3, batch_size=1)
oneslack = OneSlackSSVM(model, C=.1, tol=.1, verbose=3)
oneslack_cache = OneSlackSSVM(model, C=.1, tol=.1, inference_cache=50, verbose=3)

svms = [bcfw, pystructsgd, oneslack, oneslack_cache, pegasos, nslack, nslack_every]
names = ['bcfw', 'pystructsgd_momentum', "oneslack", "oneslack_cache", "pegasos", "nslack", "nslack_every"]

print(len(X))
X_train, y_train = shuffle(X_train, y_train)

for name, svm in zip(names, svms):
    logger = SaveLogger("letters_big_" + name + ".pickle", save_every=10)
    svm.logger = logger
    svm.fit(X_train, y_train)
