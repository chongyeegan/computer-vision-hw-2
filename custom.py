import numpy as np
import math

__author__ = 'jhh283'


# general feature vector function:
# Note must return a np.array in the desired format given a point in [y, x] format and the origin image
def generate_feature(point, image):
    color = np.zeros((image.shape[2],))
    feature = np.concatenate([color, point], axis=-1)
    for i in xrange(image.shape[2]):
        feature[i] = image[feature[-2], feature[-1]][i]
    return feature


# general getter function which returns [y, x]
def get_yx(feat):
    return feat[-2:]


# general distance function
# should be provided a point vector, center vector, relative weight for spatial coordinates, and a "step" size
# this implements the suggestion found in the original SLIC paper
def calculate_distance(vector, center, weight, step):
    ds = math.sqrt((center[3] - vector[3]) ** 2 + (center[4] - vector[4]) ** 2)
    dc = math.sqrt((center[0] - vector[0]) ** 2 + (center[1] - vector[1]) ** 2 + (center[2] - vector[2]) ** 2)
    d = math.sqrt((dc ** 2) + ((ds/step) ** 2)*(weight**2))
    return d
