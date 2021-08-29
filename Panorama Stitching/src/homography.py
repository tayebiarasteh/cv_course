"""
Created on May 17, 2020.
Homography

@authors:
Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de> https://github.com/tayebiarasteh/
Amin Heydarshahi <amin.heydarshahi@fau.de> https://github.com/aminheydarshahi/
"""
import numpy as np
import cv2
import pdb


def computeHomography(points1, points2):
    '''
    Compute a homography matrix from 4 point matches.

    :points1: list of 4 points (tuple)
    :points2: list of 4 points (tuple)
    '''
    assert(len(points1) == 4)
    assert(len(points2) == 4)

    # 8x9 matrix A based on the formula from the manual sheet.
    A = np.zeros((8,9))
    for i in range(len(points1)):
        A[i*2:i*2 +2] = np.array([[-points1[i][0], -points1[i][1], -1, 0,0,0,
                                   points1[i][0]*points2[i][0], points1[i][1]*points2[i][0], points2[i][0]],
                          [0,0,0, -points1[i][0], -points1[i][1], -1,
                           points1[i][0]*points2[i][1], points1[i][1]*points2[i][1], points2[i][1]]])

    # SVD decomposition on A
    U, s, V_transposed = np.linalg.svd(A, full_matrices=True)
    V = np.transpose(V_transposed)

    # homogeneous solution of Ah=0 as the rightmost column vector of V.
    H = V[:,-1].reshape(3,3)

    # Normalize H by 1/h8.
    H /= V[:,-1][-1]

    return H



def testHomography():
    '''
    A small test to validate the implementation of computeHomography().
    '''
    points1 = [(1, 1), (3, 7), (2, -5), (10, 11)]
    points2 = [(25, 156), (51, -83), (-144, 5), (345, 15)]

    H = computeHomography(points1, points2)

    print ("Testing Homography...")
    print ("Your result:" + str(H))

    Href = np.array([[-151.2372466105457,   36.67990057507507,   130.7447340624461],
                 [-27.31264543681857,   10.22762978292494,   118.0943169422209],
                 [-0.04233528054472634, -0.3101691983762523, 1]])

    print ("Reference: " + str(Href))

    error = Href - H
    e   = np.linalg.norm(error)
    print ("Error: " + str(e))

    if (e < 1e-10):
        print ("Test: SUCCESS!")
    else:
        print ("Test: FAIL!")
    print ("============================")
