import numpy as np



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-S", help="path of Sparse numpy file", type=str)
    parser.add_argument("-thresh", type=float, help="threshold between 0 and 1. above threshold set to 1. below set to zero")
    args = parser.parse_args()
    S_path = args.S
    threshold = args.thresh


    S_orig = np.abs(np.load(S_path))
    S_binary = S_orig.copy()
    S_binary[S_binary<threshold] = 0
    S_binary[S_binary>=threshold] = 1

    np.save(S_path[:-4]+'_thresh_'+ str(threshold)+'.npy', S_binary)

for ii in range(0,30,3):
    plt.imshow(S_binary[ii],cmap='gray')
    plt.show()


