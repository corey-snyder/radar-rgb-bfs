import matplotlib.pyplot as plt


def plot_func(output,D):
    n = 3
    N = len(D)//n
    plt.figure(figsize=(5,35))
    for ii in range(N):
        plt.subplot(N,3,3*ii+1)
        plt.imshow(D[n*ii,0],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Original')

        plt.subplot(N, 3, 3 * ii + 2)
        plt.imshow(output[n*ii, 0],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('L')

        plt.subplot(N, 3, 3 * ii + 3)
        plt.imshow(np.abs(output[n*ii, 1]),cmap='gray',vmin=0,vmax=1)
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('abs(S)')
    plt.tight_layout()