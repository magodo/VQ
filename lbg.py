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
import copy

def cluster_point(x, mu):
    """Cluster one point"""
    bestmukey = min([(i[0], np.linalg.norm(x-i[1])) \
            for i in enumerate(mu)], key=lambda t: t[1])[0]
    return bestmukey

def cluster_array(X, mu):
    """Go through array and cluster each point in it.
    :param X: Input array consists of n-D point.
    :param mu: List of mu.

    :return: Dictionary. Key: mu number. Value: list of n-D point belongs to this cluster.
    """
    clusters = {}
    for x in X:
        # Calculate the best mu key's index of each point
        bestmukey = cluster_point(x, mu)
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(clusters):
    """Go through the providen cluster dictionary. Calculate new mu vector"""
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def k_means(X, K, mu=None, show = False, fig=None, iter_count=[0]):
    """Perform K-means(arbitrary dimension vector).

    :param X: array of vectors.
    :param K: number of target cluster.
    :param mu: initial mu vector. If None, will generate random vector based on input array.
    :param show: If False: show nothing;
                 If True: if fig is None, show k-means process.
                          if fig is not None, show lbg process.

    :return: tuple consisting final clusters and mu vector.
    """
    if not mu:
        # Initialize to K random centers
        oldmu = np.array(random.sample(X, K))
        mu = np.array(random.sample(X, K))
    else:
        mu_vector_length = K
        mu_vector_dimens = len(mu[0])
        oldmu = np.array([np.zeros(mu_vector_dimens) for i in range(mu_vector_length)])

    # Convert X to numpy array
    X = np.array(X)

    if show and not fig:
        fig = plt.figure()
    while not has_converged(mu, oldmu):
        iter_count[0] += 1
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_array(X, mu)
        if show:
            # Show
            shower(clusters, mu, fig, iter_count[0])
        # Reevaluate centers
        mu = reevaluate_centers(clusters)

    print iter_count[0]
    return (mu, clusters)

def _increase(vector, dist):
    """Increase vector to a random direction with distence dist."""

    # Just apply the distence to a random dimension
    random_index = random.sample(range(len(vector)), 1)[0]
    vector[random_index] += dist
    return vector

def lbg(X, M, show = False):
    """Perform LBG algorithm.

    :param X: array of vectors.
    :param M: number of target cluster.
    :param show: If or not show the process.

    :return: tuple consisting final clusters and mu vector."""

    # First iteration
    k = 1
    mu = [np.mean(X, axis=0)]

    # Evaluate each increment distance
    dist = np.mean([np.linalg.norm(mu[0]-i) for i in X]) / 10

    if show:
        fig = plt.figure()
    else:
        fig = None

    iter_count = [0]
    while k < M:
        seperated_mu = copy.deepcopy(mu)

        if  2*k < M:
            k = 2 * k
            for i in mu:
                seperated_mu.append(_increase(copy.deepcopy(i), dist))
            #print seperated_mu, k

        else:
            for i in random.sample(range(len(mu)), M-k):
                seperated_mu.append(_increase(copy.deepcopy(mu[i]), dist))
            k = M
        mu, clusters = k_means(X, k, seperated_mu, show = show, fig = fig, iter_count=iter_count)
    #print seperated_mu, k
    return (mu, clusters)

#~~~~~~~~~~~
#Shower
#~~~~~~~~~~~
def shower(clusters, mu, fig, iter_count):

    # So K should be less than 7 if needed to be shown
    __COLORS__ = ['b', 'c', 'g', 'k', 'm', 'r', 'y']

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

    plt.savefig("lbg-%d.png"%iter_count)
    fig.show()
    plt.pause(0.1)


#~~~~~~~~~~~
#Init plane
#~~~~~~~~~~~

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



if __name__ == "__main__":
    N = 200
    X = init_plane(N)
    K = 7


    #mu, clusters = k_means(X, K)
    #print mu
    mu, clusters = lbg(X, K, show = True)
    print mu, K

