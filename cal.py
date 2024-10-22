from pyemd import emd
import numpy as np
import json

def calc_hist_distance(hist1, hist2, bin_edges):
    bins = np.array(bin_edges)
    bins_dist = np.abs(bins[:, None] - bins[None, :])
    hist_dist = emd(hist1, hist2, bins_dist)
    return hist_dist




with open('hist_stats1.json', 'r') as file:
    data1 = json.load(file)

with open('hist_stats_gt1.json', 'r') as file:
    data2 = json.load(file)

for k in data1['stats']:
    hist_dist = calc_hist_distance(np.array(data1['stats'][k]), np.array(data2['stats'][k]), np.array(data2['ticks'][k])[1:])
    print(f"hist_dist({k}): {hist_dist:.4f}")