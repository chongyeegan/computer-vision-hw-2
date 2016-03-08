from skimage import util, color
import numpy as np
import sys
import math

__author__ = 'jhh283'


def get_k_centers(k, height, width):
    '''
    Function that finds n regularly spaced along a matrix of size height and width.
    Takes a parameter k (desired number of centers), returns actual calculated centers (as close to k as possible) coordinates.

    Parameters:
    k - Number of clusters to try to fit
    height - height of the input matrix
    width - width of the input matrix

    Returns:
    centers - list of center coordinates in the following format [[center_y, center_x], ... ]
    step_y - the step size between centers on the y axis (height)
    step_x - the step size between centers on the x axis
    '''
    # setting up regular grid function using meshgrids
    grid_y, grid_x = np.mgrid[:height, :width]
    slices = util.regular_grid((height, width), k)
    step_y, step_x = [s.step if s.step is not None else 1 for s in slices]

    centers_y = grid_y[slices]
    centers_x = grid_x[slices]

    # reshape centers into desired output dimensions
    centers = np.concatenate([centers_y[..., np.newaxis], centers_x[..., np.newaxis]], axis=-1)
    centers = centers.reshape(-1, 2)

    return centers, step_y, step_x


def get_window_bounds(center, step_y, step_x, height, width):
    '''
    Function that finds start and end window coordinates of size 2*step_X x 2*step_y based on an
    input 'center' coordinateof the window. The function pads max axis by 1 to account for python
    iteration.

    Parameters:
    center - center coordinate of the window in the input format of [y, x]
    step_y - step size in the y dimension
    step_x - step size in the x dimension
    height - height of the input matrix
    width - width of the input matrix

    Returns:
    y_min, x_min - starting y, x coordinate
    y_max, x_max - ending y, x coordinate (padded by 1)
    '''
    # print center, step_y, step_x, height, width
    y_min = int(max(center[0] - step_y, 0))
    y_max = int(min(center[0] + step_y + 1, height))
    x_min = int(max(center[1] - step_x, 0))
    x_max = int(min(center[1] + step_x + 1, width))
    return y_min, y_max, x_min, x_max


def generate_features_vec(points, generate_feature, image):
    '''
    Helper function to generate feature vectors for a list of points.

    Parameters:
    points - a list of points to generate a feature vector for
    generate_feature - user-provided function that is used to generate desired feature vectors - must return a np.array in the desired format given a point in [y, x] format and the origin image
    image - np array that represents the image (should be size (height, width))

    Returns: np array of feature vectorized version of the points - height should be the same length as points
    '''
    feature_vec = []
    for point in points:
        feature_vec.append(generate_feature(point, image))
    return np.asarray(feature_vec)


def slic(image, k, max_iter, weight, generate_feature, calculate_distance, get_yx, distance_start=None, distance_end=None, nearest_centers=None, lab=False):
    '''
    Implementation of the SLIC algorithm. This provides 2 interfaces to running. If the user provides output data structures, those data structures will be used and modified.

    Parameters:
    image - np array that represents the image (should be size (height, width))
    k - desired number of centers/segments
    weight - desired weight value for the spatial coordinates
    generate_feature - user-provided function that is used to generate desired feature vectors - must return a np.array in the desired format given a point in [y, x] format and the origin image
    generate - user-provided function that is used to calculate distance based on input feature vectors - should have parameters for point vector, center vector, relative weight for spatial coordinates, and a "step" size
    get_yx - user-provided function that is used to retrieve [y, x] from the input feature vector
    distance_start (optional) - initialized np matrix of size (height, width). used to track starting distance.
    distance_end (optional) - initialized np matrix of size (height, width). used to track ending distance.
    nearest_centers (optional) - initialized np matrix of size (height, width). used to track segmentation result.

    Returns:
    nearest_centers - np array of size (height, width) where each element is assigned a 'center' identifier -- the output segmentation
    distance_start - np array of size (height, width) where each element is the initial distance calculated from the initial initialization of centers
    distance_end - np array of size (height, width) where each element is the ending distance calculated from the final iteration of centers
    '''
    image_color = np.copy(image)
    if lab is True:
        image_color = color.rgb2lab(image_color)
    height, width = image.shape[:2]

    # get initial center assignments
    centers_yx, step_y, step_x = get_k_centers(k, height, width)
    n_centers = centers_yx.shape[0]

    # print out the actual number of centers that the algorithm returns
    print 'centers', n_centers

    # implements the initialization window size described in the homework
    S_ = math.sqrt(float(height*width)/n_centers)
    step_y = int(S_)
    step_x = int(S_)

    step = max(step_y, step_x)
    print 'step size', step

    if distance_start is None:
        distance_start = np.empty((height, width), dtype=np.double)

    if distance_end is None:
        distance_end = np.empty((height, width), dtype=np.double)

    if nearest_centers is None:
        nearest_centers = np.empty((height, width), dtype=np.intp)

    distance = np.empty((height, width), dtype=np.double)
    n_center_elems = np.zeros(n_centers, dtype=np.intp)

    # generate features for all initial centers - this structure is used to track distances throughout execution
    c_feat_all = generate_features_vec(centers_yx, generate_feature, image_color)

    for i in xrange(max_iter):
        change = False
        distance[:, :] = sys.float_info.max
        # for each center, calculate window -- iterate through window and assign each window pixel a center based on the closest center available
        for k in xrange(n_centers):
            c_feat = c_feat_all[k]
            c = get_yx(c_feat)
            y_min, y_max, x_min, x_max = get_window_bounds(c, step_y, step_x, height, width)
            for y in xrange(y_min, y_max):
                for x in xrange(x_min, x_max):
                    yx_feat = generate_feature((y, x), image_color)
                    dist = calculate_distance(yx_feat, c_feat, weight, step)
                    # this performs comparison of distance for each center
                    if dist < distance[y, x]:
                        nearest_centers[y, x] = k
                        distance[y, x] = dist
                        change = True

        # store distances at beginning and end
        if i == 0:
            distance_start = np.copy(distance)
        elif i == max_iter-1:
            distance_end = np.copy(distance)
            break

        # break if converged
        if change is False:
            break

        # recalculate centers by average algorithm
        n_center_elems[:] = 0
        c_feat_all[:, :] = 0
        for y in xrange(height):
            for x in xrange(width):
                n_center_elems[nearest_centers[y, x]] += 1
                yx_feat = generate_feature((y, x), image_color)
                c_feat_all[nearest_centers[y, x]] += yx_feat

        for k in xrange(n_centers):
            c_feat_all[k] /= n_center_elems[k]

    return nearest_centers, distance_start, distance_end
