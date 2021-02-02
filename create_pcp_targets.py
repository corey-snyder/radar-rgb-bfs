from pcp import pcp
import numpy as np
import argparse
from pathlib import Path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="path of input file", type=str)
    parser.add_argument("-iterations", help="max number of iteration for PCP", type=int)

    args = parser.parse_args()
    input_path = args.input
    n_iterations = args.iterations
    output_dir = str(Path(input_path).parent)

    D = np.load(input_path)
    n_frames, im_height, im_width = D.shape
    D = D.reshape(n_frames,-1).T
    L, S, (u, s, v), errors, ranks, nnzs = pcp(D,maxiter=n_iterations)
    L = L.T.reshape(n_frames,im_height,im_width)
    S = S.T.reshape(n_frames,im_height,im_width)
    np.save(output_dir + '/S_test.npy',S)
    np.save(output_dir + '/L_test.npy',L)

    pass
    # yaml will aslo have # iterations
    # then look at notebook for saving
    # learn vim