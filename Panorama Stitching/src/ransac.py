"""
Created on May 19, 2020.
RANSAC algorithm.

@authors:
Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de> https://github.com/tayebiarasteh/
Amin Heydarshahi <amin.heydarshahi@fau.de> https://github.com/aminheydarshahi/
"""

import cv2
import numpy as np
from homography import computeHomography
import pdb


def numInliers(points1, points2, H, threshold):
    '''
    Computes the number of inliers for the given homography.
    - Project the image points from image 1 to image 2
    - A point is an inlier if the distance between the projected point and
        the point in image 2 is smaller than threshold.
    '''
    inlierCount = 0
    ## Hint: Construct a Homogeneous point of type 'Vec3' before applying H.
    points1_homog = np.vstack((points1, np.ones((1, points1.shape[1]))))
    points2_homog = np.vstack((points2, np.ones((1, points2.shape[1]))))
    points2_estimate_homog = H @ points1_homog
    points2_estimate = points2_estimate_homog / points2_estimate_homog[-1, :]
    distance_vector = np.sqrt(np.sum((points2_estimate - points2_homog) ** 2, axis=0))
    inlierCount = np.sum(distance_vector < threshold)
    return inlierCount


def computeHomographyRansac(img1, img2, matches, iterations, threshold):
    '''
    RANSAC algorithm.
    '''
    points1 = []
    points2 = []
    for i in range(len(matches)):
        points1.append(img1['keypoints'][matches[i].queryIdx].pt)
        points2.append(img2['keypoints'][matches[i].trainIdx].pt)

    bestInlierCount = 0
    for i in range(iterations):
        subset1 = []
        subset2 = []

        # Construct the subsets by randomly choosing 4 matches.
        for _ in range(4):
            idx = np.random.randint(0, len(points1) - 1)
            subset1.append(points1[idx])
            subset2.append(points2[idx])
        # Compute the homography for this subset
        H = computeHomography(subset1, subset2)

        # Compute the number of inliers
        inlierCount = numInliers(np.array(points1).T, np.array(points2).T, H, threshold)

        # Keep track of the best homography (use the variables bestH and bestInlierCount)
        if inlierCount > bestInlierCount:
            bestInlierCount = inlierCount
            bestH = H
    print("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(bestInlierCount) + " RANSAC inliers.")
    return bestH
