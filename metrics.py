import numpy as np
from skimage import io, segmentation


# boundary recall
def RecallThickened(edge_path, GT_path):
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

    # print score, totalGT, totalImg
    return float(score) / totalGT


# Under-Segmentation
def UnderSegmentation(seg_path, GT_path):
    img = io.imread(seg_path).astype(int)
    GT = io.imread(GT_path).astype(int)

    N = GT.shape[0] * GT.shape[1]

    intersection = segmentation.join_segmentations(img, GT)
    intersection_u = np.unique(intersection)
    intersection_map = {}
    for intersect in intersection_u:
        intersect_coors = np.where(intersection == intersect)
        intersection_map[intersect] = len(intersect_coors[0])

    img_u = np.unique(img)

    score = 0
    for segment in img_u:
        im_coors = np.where(img == segment)
        P = len(im_coors[0])
        check = {}
        for i in xrange(len(im_coors[0])):
            found = intersection[im_coors[0][i], im_coors[1][i]]
            try:
                check[found]
            except KeyError:
                check[found] = True
                subscore = P - intersection_map[found]
                score += min(intersection_map[found], subscore)
    return float(score) / N
