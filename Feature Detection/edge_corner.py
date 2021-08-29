"""
Created on May 9, 2020.

@authors:
Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de> https://github.com/tayebiarasteh/
Amin Heydarshahi <amin.heydarshahi@fau.de> https://github.com/aminheydarshahi/
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy.signal import argrelextrema


def show(name, img, x, y):
    windowStartX = 10
    windowStartY = 50
    windowXoffset = 5
    windowYoffset = 40

    w = img.shape[0] + windowXoffset
    h = img.shape[1] + windowYoffset

    cv2.namedWindow(name)
    cv2.moveWindow(name, windowStartX + w * x, windowStartY + h * y)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def harrisResponseImage(img):
    '''
    Compute the spatial derivatives in x and y direction.

    :param img: input image
    :return: Harris response of the image
    '''
    dIdx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dIdy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    show("dI/dx", abs(dIdx), 1, 0)
    show("dI/dy", abs(dIdy), 2, 0)

    ##########################################################
    ## Compute Ixx, Iyy, and Ixy with
    ## Ixx = (dI/dx) * (dI/dx),
    ## Iyy = (dI/dy) * (dI/dy),
    ## Ixy = (dI/dx) * (dI/dy).
    ## Note: The multiplication between the images is element-wise (not a matrix
    ## multiplication)!!

    Ixx = dIdx ** 2
    Iyy = dIdy ** 2
    Ixy = dIdx * dIdy
    show("Ixx", abs(Ixx), 0, 1)
    show("Iyy", abs(Iyy), 1, 1)
    show("Ixy", abs(Ixy), 2, 1)

    ##########################################################
    ## Compute the images A,B, and C by blurring the
    ## images Ixx, Iyy, and Ixy with a
    ## Gaussian filter of size 3x3 and standard deviation of 1.

    kernelSize = (3, 3)
    sdev = 1
    A = cv2.GaussianBlur(Ixx, kernelSize, sdev)
    B = cv2.GaussianBlur(Iyy, kernelSize, sdev)
    C = cv2.GaussianBlur(Ixy, kernelSize, sdev)

    show("A", abs(A) * 5, 0, 1)
    show("B", abs(B) * 5, 1, 1)
    show("C", abs(C) * 5, 2, 1)

    ##########################################################
    ## Compute the harris response with the following formula:
    ## R = Det - k * Trace*Trace
    ## Det = A * B - C * C
    ## Trace = A + B
    k = 0.06
    trace = A + B
    det = A * B - C * C
    response = det - k * (trace ** 2)

    ## Normalize the response image
    dbg = (response - np.min(response)) / (np.max(response) - np.min(response))
    dbg = dbg.astype(np.float32)
    show("Harris Response", dbg, 0, 2)

    ##########################################################
    cv2.imwrite("dIdx.png", (abs(dIdx) * 255.0))
    cv2.imwrite("dIdy.png", (abs(dIdy) * 255.0))

    cv2.imwrite("A.png", (abs(A) * 5 * 255.0))
    cv2.imwrite("B.png", (abs(B) * 5 * 255.0))
    cv2.imwrite("C.png", (abs(C) * 5 * 255.0))

    cv2.imwrite("response.png", np.uint8(dbg * 255.0))

    return response


def harrisKeypoints(response, threshold=0.1):
    '''
    Generate a keypoint for a pixel,
    if the response is larger than the threshold
    and it is a local maximum.
    Don't generate keypoints at the image border.

    Note 1: Keypoints are stored with (x,y) and images are accessed with (y,x)!!

    Note 2: with changing k in the R equation, we detect different number of corners.
        k = 0.005 is the best according to this image.

    :param response: Harris response of an image
    :param threshold: Minimum intensity of peaks
    :return: list of the keypoints
    '''
    points = []
    maxima = peak_local_max(response, min_distance=1, threshold_abs=threshold)
    for maximum in maxima:
        points.append(cv2.KeyPoint(maximum[1], maximum[0], 1))

    return points


def harrisEdges(input, response, edge_threshold=-0.01):
    '''
    Set edge pixels to red.

    A pixel belongs to an edge, if the response is smaller than a threshold
    and it is a minimum in x or y direction.
    Don't generate edges at the image border.
    :param input: input image
    :param response: harris response of the image
    :param edge_threshold: Maximum intensity
    '''
    result = input.copy()

    # 1) if the response is smaller than a threshold
    response = np.where(response > edge_threshold, np.inf, response)

    # 2) it is a minimum in x or y direction
    for x, y in zip(range(response.shape[0]), range(response.shape[1])):
        minima_x = argrelextrema(response[:, y], np.less)
        minima_y = argrelextrema(response[x], np.less)
        result[minima_x, x] =  (0, 0, 255)
        result[y, minima_y] =  (0, 0, 255)

    return result


def main():
    input_img = cv2.imread('blox.jpg')  ## read the image
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  ## convert to grayscale
    input_gray = (input_gray - np.min(input_gray)) / (np.max(input_gray) - np.min(input_gray))  ## normalize
    input_gray = input_gray.astype(np.float32)  ## convert to float32 for filtering

    ## Obtain Harris Response, corners and edges
    response = harrisResponseImage(input_gray)
    points = harrisKeypoints(response)
    edges = harrisEdges(input_img, response)

    imgKeypoints1 = cv2.drawKeypoints(input_img, points, outImage=None, color=(0, 255, 0))
    show("Harris Keypoints", imgKeypoints1, 1, 2)
    show("Harris Edges", edges, 2, 2)

    cv2.imwrite("edges.png", edges)
    cv2.imwrite("corners.png", imgKeypoints1)


if __name__ == '__main__':
    main()
