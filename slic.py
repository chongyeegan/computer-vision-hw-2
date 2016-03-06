from skimage import util
import numpy as np
import sys
import math


def get_k_centers(k, height, width):
    grid_y, grid_x = np.mgrid[:height, :width]
    slices = util.regular_grid((height, width), k)
    step_y, step_x = [s.step if s.step is not None else 1 for s in slices]

    centers_y = grid_y[slices]
    centers_x = grid_x[slices]

    centers = np.concatenate([centers_y[..., np.newaxis], centers_x[..., np.newaxis]], axis=-1)
    centers = centers.reshape(-1, 2)

    return centers, step_y, step_x


def get_window_bounds(center, step_y, step_x, height, width):
    # print center, step_y, step_x, height, width
    y_min = int(max(center[0] - step_y, 0))
    y_max = int(min(center[0] + step_y + 1, height))
    x_min = int(max(center[1] - step_x, 0))
    x_max = int(min(center[1] + step_x + 1, width))
    return y_min, y_max, x_min, x_max


def generate_features_vec(points, generate_feature, image):
    feature_vec = []
    for point in points:
        feature_vec.append(generate_feature(point, image))
    return np.asarray(feature_vec)


def slic(image, k, max_iter, weight, generate_feature, calculate_distance, get_yx):
    height, width = image.shape[:2]
    centers_yx, step_y, step_x = get_k_centers(k, height, width)

    n_centers = centers_yx.shape[0]
    # print centers_yx
    print 'centers', n_centers

    # TODO: confirm this change
    S_ = math.sqrt(float(height*width)/n_centers)
    step_y = int(S_)
    step_x = int(S_)

    step = max(step_y, step_x)
    print 'step size', step

    distance_start = np.empty((height, width), dtype=np.double)
    distance_end = np.empty((height, width), dtype=np.double)
    nearest_centers = np.empty((height, width), dtype=np.intp)
    distance = np.empty((height, width), dtype=np.double)
    n_center_elems = np.zeros(n_centers, dtype=np.intp)

    c_feat_all = generate_features_vec(centers_yx, generate_feature, image)

    for i in xrange(max_iter):
        change = False
        distance[:, :] = sys.float_info.max
        for k in xrange(n_centers):
            # print c_feat_all
            c_feat = c_feat_all[k]
            c = get_yx(c_feat)
            y_min, y_max, x_min, x_max = get_window_bounds(c, step_y, step_x, height, width)
            for y in xrange(y_min, y_max):
                for x in xrange(x_min, x_max):
                    yx_feat = generate_feature((y, x), image)
                    dist = calculate_distance(yx_feat, c_feat, weight, step)
                    if dist < distance[y, x]:
                        nearest_centers[y, x] = k
                        distance[y, x] = dist
                        change = True

        if i == 0:
            distance_start = np.copy(distance)
        elif i == max_iter-1:
            distance_end = np.copy(distance)
            break

        if change is False:
            break

        # TODO: maybe refactor this part
        n_center_elems[:] = 0
        c_feat_all[:, :] = 0
        for y in xrange(height):
            for x in xrange(width):
                n_center_elems[nearest_centers[y, x]] += 1
                yx_feat = generate_feature((y, x), image)
                c_feat_all[nearest_centers[y, x]] += yx_feat

        for k in xrange(n_centers):
            c_feat_all[k] /= n_center_elems[k]

    return nearest_centers, distance_start, distance_end
