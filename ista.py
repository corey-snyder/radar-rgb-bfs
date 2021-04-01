# -*- coding: utf-8 -*-
"""

"""

from __future__ import division, print_function

__all__ = ["pcp"]

import time
import fbpca
import logging
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from pcp import _svd, shrink


def ista(M, delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, **svd_args):
    # Check the SVD method.
    allowed_methods = ["approximate", "exact", "sparse"]
    if svd_method not in allowed_methods:
        raise ValueError("'svd_method' must be one of: {0}"
                         .format(allowed_methods))

    # Check for missing data.
    shape = M.shape
    if missing_data:
        missing = ~(np.isfinite(M))
        if np.any(missing):
            M = np.array(M)
            M[missing] = 0.0
    else:
        missing = np.zeros_like(M, dtype=bool)
        if not np.all(np.isfinite(M)):
            logging.warn("The matrix has non-finite entries. "
                         "SVD will probably fail.")

    # Initialize the tuning parameters.
    lam = 1.0 / np.sqrt(np.max(shape))
    if mu is None:
        mu = 0.25 * np.prod(shape) / np.sum(np.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    S = np.zeros(shape)
    L = np.zeros(shape)
    errors = []
    ranks = []
    nnzs = []
    while i < max(maxiter, 1):

        # compute L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, M - S, rank+1, 1./mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(s), v))

        # Compute S
        # Shrinkage step.
        S = shrink(M - L, lam / mu)

        step = M - L - S

        # Check for convergence.
        err = np.sqrt(np.sum(step ** 2) / norm)
        errors.append(err)
        ranks.append(np.sum(s > 0))
        nnzs.append(np.sum(S > 0))

        if verbose:
            print(("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:d}, "
                   "time={4:.3e}")
                  .format(i, err, np.sum(s > 0), np.sum(S > 0), svd_time))
            if (plot_num is not None) and (i % plot_num ==0):
                if im_shape is None:
                    print('Need im shape')
                    continue
                plt.figure()
                plt.subplot(211)
                plt.imshow(L[:,0].reshape(im_shape),cmap='gray')
                plt.subplot(212)
                plt.imshow(np.abs(S[:,0]).reshape(im_shape),vmin=0,vmax=1,cmap='gray')
                plt.show()

        if err < delta:
            break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return L, S, (u, s, v), errors, ranks, nnzs


def radar_ista(M, M_F, delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, **svd_args):
    # Check the SVD method.
    allowed_methods = ["approximate", "exact", "sparse"]
    if svd_method not in allowed_methods:
        raise ValueError("'svd_method' must be one of: {0}"
                         .format(allowed_methods))

    # Check for missing data.
    shape = M.shape
    if missing_data:
        missing = ~(np.isfinite(M))
        if np.any(missing):
            M = np.array(M)
            M[missing] = 0.0
    else:
        missing = np.zeros_like(M, dtype=bool)
        if not np.all(np.isfinite(M)):
            logging.warn("The matrix has non-finite entries. "
                         "SVD will probably fail.")

    # Initialize the tuning parameters.
    lam = 1.0 / np.sqrt(np.max(shape))
    if mu is None:
        mu = 0.25 * np.prod(shape) / np.sum(np.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    S = np.zeros(shape)
    L = np.zeros(shape)
    errors = []
    ranks = []
    nnzs = []
    while i < max(maxiter, 1):

        # compute L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, M - S, rank+1, 1./mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(s), v))

        # Compute S
        # Shrinkage step.
        S = shrink(M - L, M_F * lam / mu)

        step = M - L - S

        # Check for convergence.
        err = np.sqrt(np.sum(step ** 2) / norm)
        errors.append(err)
        ranks.append(np.sum(s > 0))
        nnzs.append(np.sum(S > 0))

        if verbose:
            print(("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:d}, "
                   "time={4:.3e}")
                  .format(i, err, np.sum(s > 0), np.sum(S > 0), svd_time))
            if (plot_num is not None) and (i % plot_num ==0):
                if im_shape is None:
                    print('Need im shape')
                    continue
                plt.figure()
                plt.subplot(211)
                plt.imshow(L[:,0].reshape(im_shape),cmap='gray')
                plt.subplot(212)
                plt.imshow(np.abs(S[:,0]).reshape(im_shape),vmin=0,vmax=1,cmap='gray')
                plt.show()

        if err < delta:
            break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return L, S, (u, s, v), errors, ranks, nnzs

def joint_ista(M,R,delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, **svd_args):
    pass

if __name__ == '__main__':
    from skimage.io import imread
    import os
    import pickle


    image_dir = 'RGB_images'
    bounds_file_path = 'bounds_vel_thresh.txt'
    # bounds_file_path = 'bounds.txt'
    images = np.array([imread(os.path.join(image_dir, 'rgb_{}.png'.format(i)), 'gray') for i in range(50)])
    D = images.reshape(images.shape[0], -1).T

    L_ista, S_ista, _, errors, ranks, nnzs = ista(D, verbose=True, maxiter=400,
                                                  plot_num=10, im_shape=[720, 1280])