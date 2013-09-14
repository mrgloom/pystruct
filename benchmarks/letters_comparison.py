import numpy as np
import matplotlib.pyplot as plt

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM

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
pegasos1 = SubgradientSSVM(model=model, C=.1, max_iter=100, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=len(X_train) * .1)
pegasos2 = SubgradientSSVM(model=model, C=.1, max_iter=20, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=100)
pegasos3 = SubgradientSSVM(model=model, C=.1, max_iter=20, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=10)
pegasos4 = SubgradientSSVM(model=model, C=.1, max_iter=20, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=0.1)
pystructsgd = SubgradientSSVM(model=model, C=.1, max_iter=100, verbose=3)

print(len(X))
X_train, y_train = shuffle(X_train, y_train)

pegasos1.fit(X_train, y_train)
#pegasos2.fit(X_train, y_train)
#pegasos3.fit(X_train, y_train)
#pegasos4.fit(X_train, y_train)
pystructsgd.fit(X_train, y_train)
bcfw.fit(X_train, y_train)

iterations = np.arange(len(bcfw.objective_curve_)) * bcfw.check_dual_every
plt.plot(iterations, bcfw.objective_curve_, label="bcfw")
plt.plot(iterations, bcfw.primal_objective_curve_, label='bcfw primal')
plt.plot(pegasos1.objective_curve_, label="pegasos1")
#plt.plot(pegasos2.objective_curve_, label="pegasos2")
#plt.plot(pegasos3.objective_curve_, label="pegasos3")
#plt.plot(pegasos4.objective_curve_, label="pegasos3")
plt.plot(pystructsgd.objective_curve_, label="pystruct defaults")
plt.legend()
plt.show()

