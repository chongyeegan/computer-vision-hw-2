weight = 270
k = 16 # k

%time nearest_centers, distance_start, distance_end = slic(image, \
                                                           k, \
                                                           max_iter, \
                                                           weight, \
                                                           generate_feature, \
                                                           calculate_distance, \
                                                           get_yx)

marked = segmentation.mark_boundaries(image, nearest_centers, mode='outer')
bounds = segmentation.find_boundaries(nearest_centers, mode='outer').astype(int)

min_dist = min(np.min(distance_start), np.min(distance_end))
max_dist = min(np.max(distance_start), np.max(distance_end))
print 'distances:', min_dist, max_dist

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
ax.imshow(marked)

fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
heatmap = ax2.imshow(distance_start, vmin=min_dist, vmax=max_dist)
fig2.colorbar(heatmap)

fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
heatmap2 = ax3.imshow(distance_end, vmin=min_dist, vmax=max_dist)
fig3.colorbar(heatmap2)


