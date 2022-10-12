from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import scipy.linalg
import cv2
import os


def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    N = scipy.linalg.null_space(F)
    Nt = scipy.linalg.null_space(F.T)
    e1 = np.array([N[0]/N[2], N[1]/N[2]])
    e2 = np.array([Nt[0]/Nt[2], Nt[1]/Nt[2]])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera intrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera intrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    pcd = None
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    E = K2.T@F@K1
    print(E)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    M1 = K1 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).astype(float)
    c1 = np.hstack((R1, t))
    c2 = np.hstack((R1, -1*t))
    c3 = np.hstack((R2, t))
    c4 = np.hstack((R2, -1*t))
    combs = np.array([c1,c2,c3,c4])
    ind = 0
    maxPts = -1
    for x in range(4):
        tri = cv2.triangulatePoints(M1, K2@combs[x], pts1.T, pts2.T)
        pos = 0
        for y in range(tri.shape[1]):
            if tri[3,y] > 0:
                pos+=1
        if pos > maxPts:
            ind = x
            maxPts = pos
    pcd = cv2.triangulatePoints(M1, K2@combs[ind], pts1.T, pts2.T)
    pcd = dehomogenize(pcd.T)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        #######################################################################
        # TODO: Your code here                                                #
        #######################################################################
        F = find_fundamental_matrix(shape, pts1, pts2)
        if name == "reallyInwards" or name == "xtrans":
            print(compute_epipoles(F))
        #draw_epipolar(img1, img2, find_fundamental_matrix(shape, pts1, pts2), pts1, pts2)
        if name == "reallyInwards":
            
            pts = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pts)
