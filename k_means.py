#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
# Author: Zhaoting Weng
# Created Time: Tue 10 Mar 2015 07:34:05 PM CST
# File Name: simple_K_means.py
# Description:
#########################################################################

import numpy as np
import matplotlib.pyplot as plt
import random
import time

# So K should be less than 7 if needed to be shown
__COLORS__ = ['b', 'c', 'g', 'k', 'm', 'r', 'y']

def cluster_points(X, mu):
    clusters = {}
    for x in X:
        # Calculate the best mu key's index of each point
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                for i in enumerate(mu)], key=lambda t: t[1])[0]

        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K, show=True):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    iter_count = 0
    if show:
        fig = plt.figure()
    while not has_converged(mu, oldmu):
        iter_count += 1
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)

        # Show
        if show:
            shower(clusters, mu, fig, iter_count)

        # Reevaluate centers
        mu = reevaluate_centers(clusters)

    print iter_count
    return (mu, clusters)

def init_plane(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def init_plane_gauss(N, K):
    n = float(N) / k
    X = []
    for i in range(K):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range[-1, 1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def shower(clusters, mu, fig, iter_count):

    # K should be less than 7
    n = len(clusters.keys())
    if n > 7:
        raise ValueError("shower: K must be less than 7!")

    fig.clear()
    ax = fig.add_subplot(111)

    for i in range(n):

        c = __COLORS__[i]

        ax.plot(mu[i][0], mu[i][1], color=c, marker="o", markersize=10)

        for x in clusters[i]:
            ax.plot(x[0], x[1], color=c, marker="*", markersize=6)

    fig.show()
    plt.savefig("k-means-%d.png"%iter_count)
    plt.pause(0.5)



if __name__ == "__main__":
    N = 200
    X = init_plane(N)
    K = 7
    mu, clusters = find_centers(X, K, show=True)
    print mu

