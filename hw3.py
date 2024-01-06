import numpy
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    mean = np.mean(x, axis=0)
    return x - mean


def get_covariance(dataset):
    # Your implementation goes here!
    y = np.dot(np.transpose(dataset), dataset)
    return y * 1/(len(dataset)-1)


def get_eig(S, m):
    # Your implementation goes here!
    w_, v_ = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    S = np.flip(w_)
    d = np.diagflat(S)
    L = np.fliplr(v_)
    return d, L

#x = load_and_center_dataset('YaleB_32x32.npy')
#S = get_covariance(x)
#Lambda, U = get_eig(S, 2)
#print(Lambda)
#print(U)


def get_eig_prop(S, prop):


    w_ = eigh(S, eigvals_only=True, subset_by_value=[float('-inf'),float('inf')])
    P = np.flip(w_)
    Y = np.trace(S)
    A = []
    for i in range(len(S)):
         if w_[i]/(Y) > prop:
            A.append(w_[i])

    A = np.array(A)

    A = np.sort(A)

    x_,y_ = eigh(S, subset_by_value=[A[0], float('inf')])

    C = np.flip(x_)
    D = np.diagflat(C)
    L = np.fliplr(y_)


    return D, L

#x = load_and_center_dataset('YaleB_32x32.npy')
#S = get_covariance(x)
#Lambda, U = get_eig_prop(S, 0.07)
#print(Lambda)
#print(U)



def project_image(image, U):
    # Your implementation goes here!


    A = np.dot(np.transpose(U),image)

    B = []

    for i in range(len(U)):
        B.append(A.dot(U[i]))

    return np.array(B)

#x = load_and_center_dataset('YaleB_32x32.npy')
#S = get_covariance(x)
#Lambda, U = get_eig(S,2)
#projection = project_image(x[0],U)
#print(projection)



def display_image(orig, proj):
    # Your implementation goes here!

    orig = np.transpose(orig.reshape(32,32))
    proj = np.transpose(proj.reshape(32,32))

    fig, (ax1,ax2) = plt.subplots(1, 2)

    ax1.set_title('Original')
    ax2.set_title('Projection')

    x1 = ax1.imshow(orig,aspect='equal')
    x2 = ax2.imshow(proj,aspect='equal')

    fig.colorbar(x1, ax=ax1)
    fig.colorbar(x2, ax=ax2)


    plt.show()





#x = load_and_center_dataset('YaleB_32x32.npy')
#S = get_covariance(x)
#Lambda, U = get_eig(S, 2)
#projection = project_image(x[0], U)
#display_image(x[0], projection)






