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


def ista(M, delta=1e-6, mu=None, maxiter=400, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, im_idx=0, **svd_args):
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
                plt.subplot(121)
                plt.imshow(L[:,im_idx].reshape(im_shape),cmap='gray')
                plt.subplot(122)
                plt.imshow(np.abs(S[:,im_idx]).reshape(im_shape),vmin=0,vmax=1,cmap='gray')
                plt.show()

        # if err < delta:
        #     break
        i += 1

    # if i >= maxiter:
    #     logging.warn("convergence not reached in pcp")
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


def joint_sum_ista(M,R,delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, **svd_args):
    """
    :param M: image data
    :param R: radar data before background subtraction
    :param delta:
    :param mu:
    :param maxiter:
    :param verbose:
    :param missing_data:
    :param svd_method:
    :param plot_num:
    :param im_shape:
    :param svd_args:
    :return:
    """
    lam1 = .1
    lam2 = 2
    lam3 = 2
    lam4 = .1    # Check the SVD method.
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
    image_S = np.zeros(shape)
    radar_S = np.zeros(shape)
    image_L = np.zeros(shape)
    radar_L = np.zeros(shape)

    errors = []
    ranks = []
    nnzs = []
    while i < max(maxiter, 1):

        ##### compute image_L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, M - image_S + radar_L, rank+1, 1./mu, **svd_args)

        s = shrink(s, 1./mu)
        # s = shrink(s, lam4/lam2)
        rank = np.sum(s > 0.0)
        print("Image L rank: ", rank)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        image_L = np.dot(u, np.dot(np.diag(s), v)) - radar_L

        ##### compute radar_L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, R - radar_S + image_L, rank + 1, 1. / mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1. / mu)
        # s = shrink(s, lam4/ lam3)
        rank = np.sum(s > 0.0)
        print("Radar L rank: ", rank)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        radar_L = np.dot(u, np.dot(np.diag(s), v)) - image_L
        svd_time = time.time() - strt
        print("Time: ", svd_time)
        ###########################################

        # Compute image_S
        # Shrinkage step.
        image_S = shrink(M - image_L + radar_S,lam / mu) - radar_S
        # image_S = shrink(M - image_L + radar_S,lam1 / lam2) - radar_S

        # Compute radar_S
        # Shrinkage step.
        radar_S = shrink(R - radar_L + image_S, lam/mu) - image_S

        # radar_S = shrink(R - radar_L + image_S, lam1 / lam3) - image_S

        image_step = M - image_L - image_S
        radar_step = R - radar_L - radar_S

        # Check for convergence.
        # err = np.sqrt(np.sum(step ** 2) / norm)
        # errors.append(err)
        ranks.append(np.sum(s > 0))
        nnzs.append(np.sum(image_S > 0))

        if verbose:
            print("Iteration: ", i)
            if (plot_num is not None) and (i % plot_num ==0):
                if im_shape is None:
                    print('Need im shape')
                    continue
                plt.figure(figsize=(10,5))
                # plt.tight_layout()
                plt.suptitle("Iteration: "+ str(i))
                plt.subplot(221)
                plt.yticks([])
                plt.xticks([])
                plt.title("RGB L")
                plt.imshow(image_L[:,10].reshape(im_shape),cmap='gray')
                plt.subplot(222)
                plt.yticks([])
                plt.xticks([])
                plt.title("RGB S")
                plt.imshow(np.abs(image_S[:,10]).reshape(im_shape), cmap='gray')
                plt.subplot(223)
                plt.yticks([])
                plt.xticks([])
                plt.title("Radar L")
                plt.imshow(radar_L[:, 10].reshape(im_shape), cmap='gray')
                plt.subplot(224)
                plt.yticks([])
                plt.xticks([])
                plt.title("Radar S")
                plt.imshow(np.abs(radar_S[:, 10]).reshape(im_shape), cmap='gray')
                plt.show()

        # if err < delta:
        #     break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return image_L, image_S, radar_L, radar_S, (u, s, v), errors, ranks, nnzs


def joint_prod_ista(M,R,delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, **svd_args):
    """
    :param M: image data
    :param R: radar data before background subtraction
    :param delta:
    :param mu:
    :param maxiter:
    :param verbose:
    :param missing_data:
    :param svd_method:
    :param plot_num:
    :param im_shape:
    :param svd_args:
    :return:
    """
    eps =1e-15
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
    image_S = np.zeros(shape)
    radar_S = np.zeros(shape)
    image_L = np.ones(shape)
    radar_L = np.ones(shape)

    errors = []
    ranks = []
    nnzs = []
    while i < max(maxiter, 1):

        ##### compute image_L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, (M - image_S)*radar_L, rank+1, 1./mu, **svd_args)

        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        print("Image L rank: ", rank)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        image_L = np.clip(np.dot(u, np.dot(np.diag(s), v))/(radar_L+eps),-1,1)

        ##### compute radar_L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, (R - radar_S)*image_L, rank + 1, 1. / mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1. / mu)
        rank = np.sum(s > 0.0)
        print("Radar L rank: ", rank)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        radar_L = np.clip(np.dot(u, np.dot(np.diag(s), v))/(image_L+eps),-1,1)
        svd_time = time.time() - strt
        print("Time: ", svd_time)
        ###########################################

        # Compute image_S
        # Shrinkage step.
        image_S = np.clip(shrink((M - image_L)*radar_S,lam / mu)/(radar_S+eps),-1,1)

        # Compute radar_S
        # Shrinkage step.
        radar_S = np.clip(shrink((R - radar_L)*image_S, lam / mu)/(image_S+eps),-1,1)

        image_step = M - image_L - image_S
        radar_step = R - radar_L - radar_S

        # Check for convergence.
        # err = np.sqrt(np.sum(step ** 2) / norm)
        # errors.append(err)
        ranks.append(np.sum(s > 0))
        nnzs.append(np.sum(image_S > 0))

        if verbose:
            print("Iteration: ", i)
            if (plot_num is not None) and (i % plot_num ==0):
                if im_shape is None:
                    print('Need im shape')
                    continue
                plt.figure(figsize=(10,5))
                plt.tight_layout()
                plt.suptitle("Iteration: "+ str(i))
                plt.subplot(221)
                plt.imshow(image_L[:,0].reshape(im_shape),cmap='gray')
                plt.subplot(222)
                plt.imshow(np.abs(image_S[:,0]).reshape(im_shape),vmin=0,vmax=1,cmap='gray')
                plt.subplot(223)
                plt.imshow(radar_L[:, 0].reshape(im_shape), cmap='gray')
                plt.subplot(224)
                plt.imshow(np.abs(radar_S[:, 0]).reshape(im_shape),vmin=0,vmax=1, cmap='gray')
                plt.show()

        # if err < delta:
        #     break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return image_L, image_S, radar_L, radar_S, (u, s, v), errors, ranks, nnzs

def joint_mix_ista(M,R,delta=1e-6, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", plot_num=None, im_shape=None, **svd_args):
    """
    :param M: image data
    :param R: radar data before background subtraction
    :param delta:
    :param mu:
    :param maxiter:
    :param verbose:
    :param missing_data:
    :param svd_method:
    :param plot_num:
    :param im_shape:
    :param svd_args:
    :return:
    """
    eps =1e-15
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
    image_S = np.ones(shape)
    radar_S = np.ones(shape)
    image_L = np.zeros(shape)
    radar_L = np.zeros(shape)

    errors = []
    ranks = []
    nnzs = []
    while i < max(maxiter, 1):

        ##### compute image_L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, M - image_S + radar_L, rank + 1, 1. / mu, **svd_args)

        s = shrink(s, 1. / mu)
        rank = np.sum(s > 0.0)
        print("Image L rank: ", rank)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        image_L = np.dot(u, np.dot(np.diag(s), v)) - radar_L

        ##### compute radar_L
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, R - radar_S + image_L, rank + 1, 1. / mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1. / mu)
        rank = np.sum(s > 0.0)
        print("Radar L rank: ", rank)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        radar_L = np.dot(u, np.dot(np.diag(s), v)) - image_L
        svd_time = time.time() - strt
        print("Time: ", svd_time)
        ###########################################

        # Compute image_S
        # Shrinkage step.
        image_S = np.clip(shrink((M - image_L)*radar_S,lam / mu)/(radar_S+eps),-1,1)

        # Compute radar_S
        # Shrinkage step.
        radar_S = np.clip(shrink((R - radar_L)*image_S, lam / mu)/(image_S+eps),-1,1)

        image_step = M - image_L - image_S
        radar_step = R - radar_L - radar_S

        # Check for convergence.
        # err = np.sqrt(np.sum(step ** 2) / norm)
        # errors.append(err)
        ranks.append(np.sum(s > 0))
        nnzs.append(np.sum(image_S > 0))

        if verbose:
            print("Iteration: ", i)
            if (plot_num is not None) and (i % plot_num ==0):
                if im_shape is None:
                    print('Need im shape')
                    continue
                plt.figure(figsize=(10,5))
                plt.tight_layout()
                plt.suptitle("Iteration: "+ str(i))
                plt.subplot(221)
                plt.imshow(image_L[:,0].reshape(im_shape),cmap='gray')
                plt.subplot(222)
                plt.imshow(np.abs(image_S[:,0]).reshape(im_shape),vmin=0,vmax=1,cmap='gray')
                plt.subplot(223)
                plt.imshow(radar_L[:, 0].reshape(im_shape), cmap='gray')
                plt.subplot(224)
                plt.imshow(np.abs(radar_S[:, 0]).reshape(im_shape),vmin=0,vmax=1, cmap='gray')
                plt.show()

        # if err < delta:
        #     break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return image_L, image_S, radar_L, radar_S, (u, s, v), errors, ranks, nnzs


if __name__ == '__main__':
    import sys
    sys.path.insert(1, '/home/spencer/research/OpenRadar')  # needed to import openradar not in folder
    try: sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')  # fixes issues with env variables
    except: pass
    import numpy as np
    import mmwave as mm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pickle
    import os
    import copy
    from scipy.ndimage import gaussian_filter1d
    from scipy import interpolate
    from sklearn.preprocessing import minmax_scale


    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    radar_frames_path = 'radar_frames_csl_lobby_700.txt'
    cfg_path = '/home/spencer/research/ti-radar-hardware/mmWave/scripts/configs/14xx/indoor_human_rcs.cfg'  # radar config
    rgb_dir_path = '/home/spencer/research/radar-rgb-bfs/RGB_images'
    camera_matrix_path = 'camera_mat.npy'

    radar_frames = pickle.load(open(radar_frames_path, 'rb'))
    iwr_cfg_cmd = mm.dataloader.cfg_list_from_cfg(cfg_path)  # this is the config sent to the radar
    iwr_cfg_dict = mm.dataloader.cfg_list_to_dict(iwr_cfg_cmd)  # this is the dictionary of config

    images = np.array([plt.imread(os.path.join(rgb_dir_path, 'rgb_{}.png'.format(i))) for i in range(len(radar_frames))])
    images = rgb2gray(images)[:,::4,::4]
    D = images.reshape(images.shape[0], -1).T

    camera_mat = np.load(camera_matrix_path)
    fx = camera_mat[0, 0]
    x_c = camera_mat[0, 2]

    print('\n%d Radar Frames and %d RGB Frames Loaded' % (len(radar_frames), len(images)))

    BINS_PROCESSED = 304  # If you want to zoom in plot, make this 120
    VIRT_ANT = 8  # virtual antennas because we use time division multiple access to simulate a larger antenna array
    DOPPLER_BINS = 32
    ANGLE_RES = 1  # 1 degree resolution
    ANGLE_RANGE = 90  # so we get 181 angle bins [0,180]
    ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
    range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED), dtype=np.complex_)
    num_vec, steering_vec = mm.dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT)  # steering vector is a(theta) in most literature

    radar_range_azs = []
    for ii in range(len(radar_frames)):
        test_radar_frame = radar_frames[ii]
        test_radar_cube = mm.dsp.range_processing(test_radar_frame)
        # mean = test_radar_cube.mean(0)
        # test_radar_cube = test_radar_cube - mean

        (test_log_doppler_cube, test_doppler_cube) = mm.dsp.doppler_processing(test_radar_cube,
                                                     num_tx_antennas=iwr_cfg_dict['numTx'], clutter_removal_enabled=False,
                                                     interleaved=False, window_type_2d=mm.dsp.utils.Window.HAMMING,
                                                     accumulate=False,phase_correction=True)

        test_doppler_cube = np.fft.fftshift(test_doppler_cube, axes=(2,))
        test_log_doppler_cube = np.fft.fftshift(test_log_doppler_cube, axes=(2,))

        beamforming_result = np.zeros([BINS_PROCESSED,ANGLE_BINS],dtype=np.complex_) # range bins x angle bins

        for jj in range(BINS_PROCESSED):
            beamforming_result[jj,:], _ = mm.dsp.aoa_capon(test_doppler_cube[jj], steering_vec)

        radar_range_azs.append(np.flip(beamforming_result,axis=1))
        # plt.figure(figsize=(15,10))
        # plt.tight_layout(True)
        # plt.subplot(111, projection='polar')
        # # # cartesian coordinates (looks like real life)
        # # beamforming_result = np.flip(beamforming_result,axis=1)  # dont know why we need this but we'll be using own polar transform in future
        # #
        # azimuths = np.radians(np.linspace(0, 180, 181))
        # zeniths = np.linspace(0, range_res*BINS_PROCESSED, BINS_PROCESSED)
        # r, theta = np.meshgrid(zeniths, azimuths)
        # values = beamforming_result.T
        # plt.pcolormesh(theta, r, np.log(np.abs(values)))
        # plt.grid()
        # plt.xlim([0,np.pi])
        # plt.show()

    radar_az_amps = [np.sum(np.abs(r_a), axis=0) for r_a in radar_range_azs]
    radar_az_log_amps = [np.log(a_a) for a_a in radar_az_amps]

    # determine FOV in deg
    right = images[0].shape[1]
    left = 0

    print('Left Min Angle', 180 / np.pi * np.arctan((left - x_c) / fx))
    print('Right Max Angle', 180 / np.pi * np.arctan((4*right - x_c) / fx))

    pixel_angles = 180 / np.pi * np.arctan((4*np.arange(right) - x_c) / fx)
    front_angles = np.arange(-90, 91)

    front_angles = np.arange(-90, 91)
    pixel_amplitudes = []

    for frame_idx in range(len(radar_az_amps)):
        radar_az_log_amp = radar_az_log_amps[frame_idx]

        f = interpolate.interp1d(front_angles, radar_az_log_amp)
        pixel_amps = f(pixel_angles) - np.min(radar_az_log_amps)
        pixel_amplitudes.append(pixel_amps)
        # plt.imshow(images[frame_idx], extent=[0, 1280, 0, 720])
        # plt.plot((pixel_amps) * 200, '--', c='lime', linewidth=4)
        # plt.show()

    M_F = np.array(pixel_amplitudes)[:, :, np.newaxis] * np.ones((images.shape[0], images.shape[2], images.shape[1]))
    M_F = np.swapaxes(M_F, 1, 2).reshape(images.shape[0], -1)
    M_F = M_F.T
    M_F /= np.max(M_F)

    image_L, image_S, radar_L, radar_S, _, errors, ranks, nnzs = joint_sum_ista(D, M_F, verbose=True, maxiter=400,
                                              plot_num=50,im_shape=[720//4,1280//4])