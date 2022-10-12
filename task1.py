import numpy as np
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - M: Numpy array of shape (3,4)

    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    pts3d = np.insert(pts3d, 3, 1, axis=1)
    A = np.zeros((2*pts2d.shape[0], 12))
    zeros = np.zeros((1, 4))
    ind = 0
    for x in np.arange(0, 2*pts2d.shape[0], 2):
        A[x, 0:4] = zeros
        A[x, 4:8] = pts3d[ind, :]
        A[x, 8:12] = -1 * pts2d[ind, 1] * pts3d[ind, :]
        A[x + 1, 0:4] = pts3d[ind, :]
        A[x + 1, 4:8] = zeros
        A[x + 1, 8:12] = -1 * pts2d[ind, 0] * pts3d[ind, :]
        ind = ind + 1



    w, v = np.linalg.eig(np.dot(A.T, A))
    h = v[:, np.argmin(w)]
    M = np.reshape(h, (3, 4))
    print(M)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return M

def compute_distance(pts2d, pts3d):
    """
    use find_projection to find matrix M, then use M to compute the average 
    distance in the image plane (i.e., pixel locations) 
    between the homogeneous points M X_i and 2D image coordinates p_i

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - float: a average distance you calculated (threshold is 0.05)

    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    M = find_projection(pts2d, pts3d)
    #pts2d = np.insert(pts2d, 2, 1, axis=1)
    pts3d = np.insert(pts3d, 3, 1, axis=1)
    sum = 0
    for x in range(pts2d.shape[0]):
        pred = np.dot(M, pts3d[x, :].T)
        pred = np.array([pred[0]/pred[2], pred[1]/pred[2]])
        sum += np.sum((pred - pts2d[x,:])**2)**.5
    sum = sum/pts2d.shape[0]
    distance = sum

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distance

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """
    
    print(compute_distance(pts2d, pts3d))