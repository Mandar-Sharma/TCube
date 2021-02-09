import numpy as np
import math
from scipy import stats

from TSLR import scikit_wrappers
from sklearn import cluster

from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim, show
from matplotlib.lines import Line2D

sign = lambda x: math.copysign(1, x)

def best_line(remaining, max_error):
    test_points = remaining[:2]
    for point in remaining[2:]:
        test_points = np.append(test_points, point)
        residuals = list(np.polyfit(range(len(test_points)),test_points,deg=1,full=True)[1])
        error = 0 if not residuals else residuals[0]
        if error >= max_error:
            return test_points
    return test_points

def leastsquareslinefit(sequence,seq_range):
    x = np.arange(seq_range[0],seq_range[1]+1)
    y = np.array(sequence[seq_range[0]:seq_range[1]+1])
    A = np.ones((len(x),2),float)
    A[:,0] = x
    (p,residuals,rank,s) = np.linalg.lstsq(A,y)
    try:
        error = residuals[0]
    except IndexError:
        error = 0.0
    return (p,error)

def sumsquared_error(sequence, segment):
    x0,y0,x1,y1 = segment
    p, error = leastsquareslinefit(sequence,(x0,x1))
    return error    

def regression(sequence, seq_range):
    p, error = leastsquareslinefit(sequence,seq_range)
    y0 = p[0]*seq_range[0] + p[1]
    y1 = p[0]*seq_range[1] + p[1]
    return (seq_range[0],y0,seq_range[1],y1)

def draw_plot(data,plot_title,color):
    plot(range(len(data)),data,alpha=0.8,color=color)
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((0,len(data)-1))

def draw_segments(segments, color):
    ax = gca()
    for idx, segment in enumerate(segments):
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]), color=color)
        ax.add_line(line)

def sliding_window(data, max_error):
    data_clone = data
    anchor = 0
    return_segments = []
    while len(data_clone[anchor:]) > 0:
        i = 1
        error = 0
        while (error < max_error) and (anchor+i) < len(data):
            i += 1
            data_points = data_clone[anchor:anchor+i]
            residuals =  list(np.polyfit(range(len(data_points)),data_points,deg=1,full=True)[1])
            error = 0 if not residuals else residuals[0]
            
        params = regression(data_points,[0,len(data_points)-1])
        return_segments.append((anchor, params[1], anchor+i-1, params[3]))
        anchor += i
    return return_segments

def bottomupsegment(sequence, create_segment, compute_error, max_error):
    segments = [create_segment(sequence, [i,i+1]) for i in range(0, len(sequence)-1)]
    mergedsegments = [create_segment(sequence,(seg1[0],seg2[2])) for seg1,seg2 in zip(segments[:-1],segments[1:])]
    mergecosts = [compute_error(sequence,segment) for segment in mergedsegments]

    
    while len(mergecosts) > 0 and min(mergecosts) < max_error:
        idx = mergecosts.index(min(mergecosts))
        
        segments[idx] = create_segment(sequence, (segments[idx][0], segments[idx+1][2]))
        del segments[idx+1]
        
        mergedsegments = [create_segment(sequence,(seg1[0],seg2[2])) for seg1,seg2 in zip(segments[:-1],segments[1:])]
        mergecosts = [compute_error(sequence,segment) for segment in mergedsegments]
    
    return segments

def swab(data, buffer_percent, bottom_up_error, best_line_error):
    return_segments = []
    buffer, remaining = np.split(data, [int(len(data)*buffer_percent)])
    upper_bound = 2 * len(buffer)
    lower_bound = int(len(buffer)/2)
    anchor = 0
    while True:
        segments_retrieved = bottomupsegment(buffer, regression, sumsquared_error, bottom_up_error)
        start, end = segments_retrieved[0][0], segments_retrieved[0][2]
        buffer = np.delete(buffer, slice(start, end))
        return_segments.append((anchor, segments_retrieved[0][1], anchor+end, segments_retrieved[0][3]))
        anchor += end
        if len(remaining) > 0:         
            
            buffer_append = best_line(remaining, best_line_error)
            possible_buffer_length = len(buffer) + len(buffer_append)
            
            if possible_buffer_length < lower_bound:
                diff = lower_bound - possible_buffer_length
                buffer = np.append(buffer, remaining[:(diff+len(buffer_append))])
                remaining = np.delete(remaining, slice(0, diff+len(buffer_append)))
            
            elif possible_buffer_length > upper_bound:
                diff = possible_buffer_length - upper_bound
                buffer = np.append(buffer, buffer_append[:len(buffer_append)-diff])
                remaining = np.delete(remaining, slice(0, len(buffer_append)-diff))
            
            else:
                buffer = np.append(buffer, buffer_append)
                remaining = np.delete(remaining, slice(0, len(buffer_append)))
                
        #Flush-out
        else:
            left_to_add = []
            start = 0
            for entry in segments_retrieved[1:]:
                left_to_add.append((start, entry[1], (entry[2] - entry[0]) + start, entry[3]))
                start += entry[2] - entry[0]
            for segment in left_to_add:
                start, end = segment[0], segment[2]
                return_segments.append((anchor+start, segment[1], anchor+end, segment[3]))
            return return_segments

            
    return return_segments

def compute_error(original_data, segmentation_results):
    num_seg = len(segmentation_results)
    total_error = 0
    total_rval = 0
    for segment in segmentation_results:
        total_error += sumsquared_error(original_data, segment)
        res = stats.linregress(range(len(segment)), segment)
        total_rval += (res.rvalue**2)
    r_sqr = float(total_rval/num_seg)
    return (total_error, num_seg, r_sqr)


def rl_error_compute(k_means_results, dataset):
    total_error = 0
    total_rval = 0
    segments = split_clusters(k_means_results.labels_,dataset)
    segments = rearrange(segments)
    total_seg = len(segments)
    for segment in segments:
        error_fit = np.polyfit(range(len(segment)),segment,deg=1,full=True)[1]
        error = 0 if not error_fit else error_fit[0]
        total_error += error
        res = stats.linregress(range(len(segment)), segment)
        total_rval += (res.rvalue**2)
    r_sqr = float(total_rval/total_seg)
    return (total_error, total_seg, r_sqr)

def rearrange(segs):
    straggler = None
    re_arrangement = []
    idx_track = -1
    for idx, seg in enumerate(segs):
        if len(seg) == 1:
            straggler = seg[0]
            idx_track = idx
        else:
            if idx == (idx_track - 1):
                re_arrangement.append(seg.append(straggler))
            else:
                re_arrangement.append(seg)
    return re_arrangement

def tslr_rep(timeseries, k=5, tolerance=1e-4, cuda=True, gpu=0):
    hyperparameters = {
        "batch_size": 1,
        "channels": 30,
        "compared_length": None,
        "depth": 10,
        "nb_steps": 100,
        "in_channels": 1,
        "kernel_size": 3,
        "penalty": None,
        "early_stopping": None,
        "lr": 0.001,
        "nb_random_samples": 10,
        "negative_penalty": 1,
        "out_channels": 160,
        "reduced_size": 80,
        "cuda": cuda,
        "gpu": gpu
    }
    
    encoder = scikit_wrappers.CausalCNNEncoderClassifier()
    encoder.set_params(**hyperparameters)

    model = 'TSLR/COVIDMODELS/'
    encoder.load_encoder(model)
    
    encoded = encoder.encode_window(np.array([[timeseries]]),1)
    embeddings = np.swapaxes(encoded[0, :, :], 0, 1)
    kmeans_results = cluster.KMeans(n_clusters=k, tol=tolerance).fit(embeddings)
    
    return (embeddings, kmeans_results)
    
def cal_slope(segment, org_data):
    start, end = segment[0], segment[2]
    sequence = org_data[start:end]
    res = stats.linregress(range(len(sequence)), sequence)
    slope = res.slope
    return slope

def rearrange_segmentation(segmented_data, org_data):
    rearranged = []
    slopes = []
    for seg in segmented_data:
        slopes.append(sign(cal_slope(seg, org_data)))
    hold_out = None
    action = False
    for idx in range(len(slopes)-1):
        if slopes[idx] == slopes[idx+1] and action == False:
            hold_out = segmented_data[idx][0]
            action = True
        elif slopes[idx] == slopes[idx+1] and action == True:
            pass
        elif slopes[idx] != slopes[idx+1]:
            if hold_out is not None:
                rearranged.append((hold_out,segmented_data[idx+1][0]))
                action = False
                hold_out = None
            else:
                rearranged.append((segmented_data[idx][0],segmented_data[idx][2]))
            
    if slopes[-2] == slopes[-1]:
        rearranged.append((hold_out,segmented_data[-1][2]))
    else:
        rearranged.append((segmented_data[-2][0],segmented_data[-2][2]))
        rearranged.append((segmented_data[-1][0],segmented_data[-1][2]))
    return rearranged

def re_segment(segmented_data, org_data):
    return_list = []
    if len(segmented_data) <= 2:
        return segmented_data
    rearranged = rearrange_segmentation(segmented_data, org_data)
    for entry in rearranged:
        params = regression(org_data, [entry[0], entry[1]])
        return_list.append(params)
    return return_list

def find_trend(filtered_data, original_data):
    return_list = []
    for entry in filtered_data:
        slope_val = cal_slope(entry, original_data)
        if slope_val > 0 and slope_val <= 0.1:
            text = "Increased"
        elif slope_val > 0.1 and slope_val <= 0.6:
            text = "Sharp Increase"
        elif slope_val > 0.6:
            text = "Exponential Increase"
        elif slope_val == 0:
            text = "Flatline"
        elif slope_val < 0 and slope_val >= -0.1:
            text="Decreased"
        elif slope_val < -0.1 and slope_val >= -0.6:
            text = "Sharp Decrease"
        elif slope_val < -0.6:
            text = "Exponential Decrease"
        elif math.isnan(slope_val):
            text = "Decreased"
        return_list.append((entry[0], entry[2], text))
    return return_list