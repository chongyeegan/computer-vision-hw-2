import numpy as np
from skimage import io, segmentation

__author__ = 'jhh283'


# boundary recall
def RecallThickened(edge_path, GT_path):
    '''
    Calculation of Boundary Recall based on two edge paths (one target, one ground truth).
    Implements the "3 pixel thickening" described in the last hw.

    Parameters:
    edge_path - path/to/edge_file
    GT_path - path/to/GroundTruth_file

    Returns:
    Boundary Recall value
    '''
    img = io.imread(edge_path).astype(int)
    GT = io.imread(GT_path).astype(int) / 255

    # thickening GT
    GT_thickened = np.lib.pad(GT, ((3, 3)), 'constant').astype(int)
    for i in xrange(len(GT)):
        for j in xrange(len(GT[i])):
            if GT[i][j] == 1:
                for k in xrange(0, 4):
                    GT_thickened[i+3-k][j+3] = 1
                    GT_thickened[i+3-k][j+3-k] = 1
                    GT_thickened[i+3-k][j+3+k] = 1
                    GT_thickened[i+3+k][j+3] = 1
                    GT_thickened[i+3+k][j+3-k] = 1
                    GT_thickened[i+3+k][j+3+k] = 1
                    GT_thickened[i+3][j+3+k] = 1
                    GT_thickened[i+3][j+3-k] = 1

    totalGT = 0
    totalImg = 0
    score = 0

    # calculate intersect, total for each
    for i in range(len(GT)):
        for j in range(len(GT[i])):
            if GT[i][j] == 1:
                totalGT += 1
            if img[i][j] == 1:
                totalImg += 1
            if GT_thickened[i+3][j+3] == 1 and img[i][j] == 1:
                score += 1

    if totalImg == 0:
        totalImg = 1

    # calculate boundary recall measure
    # print score, totalGT, totalImg
    return float(score) / totalGT


# Under-Segmentation
def UnderSegmentation(seg_path, GT_path):
    '''
    Implementation of undersegmentation measurement based on two image segmentations (one target, one ground truth)

    Parameters:
    edge_path - path/to/edge_file
    GT_path - path/to/GroundTruth_file

    Returns:
    Undersegmentation value
    '''
    img = io.imread(seg_path).astype(int)
    GT = io.imread(GT_path).astype(int)

    N = GT.shape[0] * GT.shape[1]

    # find the intersection between two segmentations (this outputs all combinations of intersects)
    intersection = segmentation.join_segmentations(img, GT)

    # find all the unique intersecting segmentations
    intersection_u = np.unique(intersection)

    # create a map of counts of elements in the intersection for each intersecting segment type
    intersection_map = {}
    for intersect in intersection_u:
        intersect_coors = np.where(intersection == intersect)
        intersection_map[intersect] = len(intersect_coors[0])

    img_u = np.unique(img)

    score = 0

    # iterate through each original segmentations
    for segment in img_u:
        im_coors = np.where(img == segment)
        # calculate |P| value
        P = len(im_coors[0])
        check = {}

        # check the overlap between original segmentation and intesect
        # calculate undersegmentation
        for i in xrange(len(im_coors[0])):
            found = intersection[im_coors[0][i], im_coors[1][i]]
            try:
                check[found]
            except KeyError:
                check[found] = True
                subscore = P - intersection_map[found]
                score += min(intersection_map[found], subscore)
    return float(score) / N
