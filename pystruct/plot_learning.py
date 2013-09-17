#!/usr/bin/python
"""
This module provides a callable for easy evaluation of stored models.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np

from pystruct.utils import SaveLogger

def halton(index, base):
       result = 0
       f = 1. / base
       i = index
       while(i > 0):
           result = result + f * (i % base)
           i = np.floor(i / base)
           f = f / base
       return result


def get_color(offset=0):
    i = 0
    while True:
        c1 = halton(i + offset, 2)
        c2 = halton(i + offset, 3)
        c3 = halton(i + offset, 5)
        i += 1
        yield [c1, c2, c3]

def main():

    parser = argparse.ArgumentParser(description='Plot learning progress for one or several SSVMs.')
    parser.add_argument('pickles', metavar='N', type=str, nargs='+',
                        help='pickle files containing SSVMs')
    parser.add_argument('--time', dest='time', action='store_const',
                        const=True, default=False, help='Plot against '
                       'wall-clock time (default: plot against iterations.)')

    args = parser.parse_args()

    ssvms = []
    for file_name in args.pickles:
        print("loading %s ..." % file_name)
        ssvms.append(SaveLogger(file_name=file_name).load())
    if np.any([hasattr(ssvm, 'loss_curve_') for ssvm in ssvms]):
        n_plots = 2
    else:
        n_plots = 1
    fig, axes = plt.subplots(1, n_plots)

    # find best dual value among all objectives
    best_dual = -np.inf
    for ssvm in ssvms:
        if hasattr(ssvm, 'dual_objective_curve_'):
            best_dual = max(best_dual, np.max(ssvm.dual_objective_curve_))
    if not np.isfinite(best_dual):
        best_dual = None


    for i, (ssvm, file_name, color) in enumerate(zip(ssvms, args.pickles, get_color(1))):
        prefix = ""
        if len(ssvms) > 1:
            prefix = file_name[:-7] + " "
        plot_learning(ssvm, axes=axes, prefix=prefix, time=args.time,
                      color=color, suboptimality=best_dual)
    plt.show()


def plot_learning(ssvm, time=True, axes=None, prefix="", color=None,
    show_caching=False, suboptimality=None):
    """Plot optimization curves and cache hits.

    Create a plot summarizing the optimization / learning process of an SSVM.
    It plots the primal and cutting plane objective (if applicable) and also
    the target loss on the training set against training time.
    For one-slack SSVMs with constraint caching, cached constraints are also
    contrasted against inference runs.

    Parameters
    -----------
    ssvm : object
        Learner to evaluate. Should work with all learners.

    time : boolean, default=True
        Whether to use wall clock time instead of iterations as the x-axis.

    prefix : string, default=""
        Prefix for legend.

    color : matplotlib color.
        Color for the plots.

    show_caching : bool, default=False
        Whether to include iterations using cached inference in 1-slack ssvm.

    suboptimality : float or None, default=None
        If a float is given, only plot primal suboptimality with respect to
        this optimum.

    Notes
    -----
    Warm-starting a model might mess up the alignment of the curves.
    So if you warm-started a model, please don't count on proper alignment
    of time, cache hits and objective.
    """
    print(ssvm)
    if hasattr(ssvm, 'base_ssvm'):
        ssvm = ssvm.base_ssvm

    inference_run = None
    ssvm.timestamps_ = np.array(ssvm.timestamps_)
    primal_objective_curve = np.array(ssvm.primal_objective_curve_)
    if suboptimality is not None:
        primal_objective_curve -= suboptimality
    if hasattr(ssvm, 'cached_constraint_') and np.any(ssvm.cached_constraint_):
        # we don't want to do this if there was no constraint caching
        inference_run = ~np.array(ssvm.cached_constraint_)
        if show_caching:
            pass
        else:
            ssvm.dual_objective_curve_ = np.array(ssvm.dual_objective_curve_)[inference_run]
            primal_objective_curve = primal_objective_curve[inference_run]
            ssvm.timestamps_ = [ssvm.timestamps_[0]] + ssvm.timestamps_[1:][inference_run]
    else:
        show_caching = False

    if hasattr(ssvm, 'iterations_'):
        # BCFW remembers when we computed the objective
        iterations = ssvm.iterations_
    elif hasattr(ssvm, 'dual_objective_curve_'):
        iterations = np.arange(len(ssvm.dual_objective_curve_))
        print("Dual Objective: %f" % ssvm.dual_objective_curve_[-1])
    else:
        iterations = np.arange(len(primal_objective_curve))
        print("Primal Objective: %f" % primal_objective_curve[-1])

    print("Iterations: %d" % (np.max(iterations) + 1))  # we count from 0
    if hasattr(ssvm, "loss_curve_"):
        n_plots = 2
    else:
        n_plots = 1
    if axes is None:
        fig, axes = plt.subplots(1, n_plots)
    if not isinstance(axes, list):
        axes = [axes]

    if time:
        inds = np.array(ssvm.timestamps_)
        inds = inds[1:] / 60.
        axes[0].set_xlabel('training time (min)')
    else:
        axes[0].set_xlabel('Passes through training data')
        inds = iterations

    axes[0].set_title("Objective")
    axes[0].set_yscale('log')
    if hasattr(ssvm, "dual_objective_curve_") and suboptimality is None:
        axes[0].plot(inds, ssvm.dual_objective_curve_, '--', label=prefix + "dual objective", color=color, linewidth=3)
    axes[0].plot(inds, primal_objective_curve,
                 label=prefix + "cached primal objective" if inference_run is not None
                 else prefix + "primal objective", color=color, linewidth=3)
    if show_caching:
        axes[0].plot(inds[inference_run],
                     primal_objective_curve[inference_run], 'o', label=prefix +
                     "primal", color=color)
    axes[0].legend(loc='best')
    if n_plots == 2:
        if time:
            axes[1].set_xlabel('training time (min)')
        else:
            axes[1].set_xlabel('Passes through training data')

        try:
            axes[1].plot(inds[::ssvm.show_loss_every], ssvm.loss_curve_, color=color)
        except:
            axes[1].plot(ssvm.loss_curve_, color=color)

        axes[1].set_title("Training Error")
        axes[1].set_yscale('log')
    return axes


if __name__ == "__main__":
    main()
