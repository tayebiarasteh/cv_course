"""
Created on May 17, 2020.
k-Nearest Neighbor Search and Outlier Removal

@authors:
Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de> https://github.com/tayebiarasteh/
Amin Heydarshahi <amin.heydarshahi@fau.de> https://github.com/aminheydarshahi/
"""

import pdb
import cv2
from utils import createMatchImage
import numpy as np



def matchknn2(descriptors1, descriptors2):
    '''
    Finds the two nearest neighbors for every descriptor in image 1
        i.e. the smallest and second smallest Hamming distance.
    Store the best match (smallest distance) in knnmatches[i][0]
        and the second best match in knnmatches[i][1].

    :descriptors1: ORB feature descriptors of the image 1
        shape: (num_features, 32)
    :descriptors2: ORB feature descriptors of the image 2
        shape: (num_features, 32)
    :return: a list of DMatch objects of nearest and second nearest neighbors
        of descriptor of image 1 in that of image 2.
    '''
    knnmatches = []
    for i in range(descriptors1.shape[0]):
        distance = []
        for ii in range(descriptors2.shape[0]):
            distance.append(cv2.norm(descriptors1[i], descriptors2[ii], cv2.NORM_HAMMING))
        distance = np.asarray(distance)
        distance_sorted = np.sort(distance)
        dm1 = cv2.DMatch(i, np.argmin(distance), np.min(distance))
        dm2 = cv2.DMatch(i, np.argwhere(distance == distance_sorted[1])[0,0], distance_sorted[1])
        knnmatches.append([dm1, dm2])

    return knnmatches


def ratioTest(knnmatches, ratio_threshold):
    '''
    Outlier Removal
    Compute the ratio between the nearest and second nearest neighbor.
    Add the nearest neighbor to the output matches if the ratio is smaller than ratio_threshold
    '''
    matches = []
    for distances in knnmatches:
        if (distances[0].distance / distances[1].distance) < ratio_threshold:
            matches.append(distances[0])
    return matches


def computeMatches(img1, img2):
    knnmatches = matchknn2(img1['descriptors'], img2['descriptors'])
    matches = ratioTest(knnmatches, 0.7)
    print ("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(len(matches)) + " matches.")
    return matches
