import numpy as np

def find_contiguous(kmeans_labels):
    associated_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow', 4: 'magenta', 5: 'black', 6: 'purple', 7: 'cyan', 8: 'pink', 9: 'orange', 10: 'grey', 11: 'fuchsia', 12: 'maroon', 13: 'navy'}
    colors = [associated_colors[l] for l in kmeans_labels.labels_]
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)
    cont_seg = continuous_clusters(segs)
    return cont_seg

def continuous_clusters(segments):
    continuous = []
    last_seen_piece = segments[0][0]
    for segment in segments:
        if len(segment) > 10:
            for piece in segment:
                continuous.append(piece)
                last_seen_piece = piece
        else:
            for piece in segment:
                continuous.append(last_seen_piece)
    return continuous

def find_contiguous_sporadic(kmeans_labels):
    associated_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow', 4: 'magenta', 5: 'black', 6: 'purple', 7: 'cyan', 8: 'pink', 9: 'orange', 10: 'grey', 11: 'fuchsia', 12: 'maroon', 13: 'navy'}
    colors = [associated_colors[l] for l in kmeans_labels.labels_]
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)
    cont_seg = continuous_clusters_sporadic(segs)
    return cont_seg

def continuous_clusters_sporadic(segments):
    continuous = []
    last_seen_piece = segments[0][0]
    for segment in segments:
        if len(segment) > 5:
            for piece in segment:
                continuous.append(piece)
                last_seen_piece = piece
        else:
            for piece in segment:
                continuous.append(last_seen_piece)
    return continuous

def find_indices(colors):
    indices = []
    prev_idx = -1
    prev_color = colors[0]
    for idx, color in enumerate(colors[1:]):
        if color != prev_color:
            indices.append((prev_idx+1, idx+1))
            prev_idx = idx
            prev_color = color
    indices.append((prev_idx+1, len(colors)))
    return indices

def find_waves(data, indices, tolerance=7):
    waves = []
    for entry in indices:
        log_std = np.log(np.std(data[entry[0]:entry[1]]))
        if abs(log_std) >= tolerance:
            waves.append((entry[0], entry[1]))
    return waves
