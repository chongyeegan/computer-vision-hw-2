# height, width = image.shape[:2]

# centers_yx, step_y, step_x = get_k_centers(k, height, width)
# step = max(step_y, step_x)


# distance_start = np.empty((height, width), dtype=np.double)
# distance_end = np.empty((height, width), dtype=np.double)


# n_centers = centers_yx.shape[0]

# nearest_centers = np.empty((height, width), dtype=np.intp)
# distance = np.empty((height, width), dtype=np.double)
# n_segment_elems = np.zeros(n_centers, dtype=np.intp)

# c_feat_all = generate_features_vec(centers_yx, generate_feature, image)

# for i in xrange(max_iter):
# # for i in xrange(1):
#     change = 0
#     distance[:, :] = sys.float_info.max
#     for k in xrange(n_centers):
#         c_feat = c_feat_all[k]
#         c = get_yx(c_feat)
#         y_min, y_max, x_min, x_max = get_window_bounds(c, step_y, step_x, height, width)
#         for y in xrange(y_min, y_max):
#             for x in xrange(x_min, x_max):
#                 yx_feat = generate_feature((y, x), image)
#                 dist = calculate_distance(yx_feat, c_feat, 40, step)
#                 if dist < distance[y, x]:
#                     nearest_centers[y, x] = k
#                     distance[y, x] = dist
#                     change = 1

#     if i == 0:
#         distance_start = np.copy(distance)
#     elif i == max_iter-1:
#         distance_end = np.copy(distance)
#         break

#     if change == 0:
#         break

#     n_segment_elems[:] = 0
#     c_feat_all[:, :] = 0
#     for y in xrange(height):
#         for x in xrange(width):
#             k = nearest_centers[y, x]
#             n_segment_elems[k] += 1
#             yx_feat = generate_feature((y, x), image)
#             c_feat_all[k] += yx_feat

#     for k in xrange(n_centers):
#         c_feat_all[k] /= n_segment_elems[k]
