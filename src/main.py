#!/usr/bin/env python3

# Importing the libraries
import os
import wget
import gzip
import time
import warnings
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
from os.path import exists
from scipy.sparse import *
from scipy.sparse.linalg import norm
import plotly.graph_objs as go

warnings.simplefilter(action='ignore', category=FutureWarning)
# some stupid pandas function that doesn't work

class Utilities:
    # Importing the dataset
    def load_data():
        # Loading the dataset
        dataset = int(input("Choose the dataset:\n  [1] web-Stanford (use this one for now)\n  [2] web-BerkStan: \nEnter an option: "))

        if dataset == 1:
            if exists('../data/web-Stanford.txt'):
                dataset = '../data/web-Stanford.txt'
            else:
                print("\nThe file doesn't exist, download it from https://snap.stanford.edu/data/web-Stanford.html")

                # if there is no folder data, create it
                if not exists('../data'):
                    os.makedirs('../data')

                # Downloading the dataset
                url = 'https://snap.stanford.edu/data/web-Stanford.txt.gz'
                wget.download(url, '../data/web-Stanford.txt.gz')
                # Unzipping the dataset
                with gzip.open('../data/web-Stanford.txt.gz', 'rb') as f_in:
                    with open('../data/web-Stanford.txt', 'wb') as f_out:
                        f_out.write(f_in.read())

                # delete the zipped file
                os.remove('../data/web-Stanford.txt.gz')

                dataset = '../data/web-Stanford.txt'
                print("\nDataset downloaded\n")

        elif dataset == 2:
            if exists('../data/web-BerkStan.txt'):
                dataset = '../data/web-BerkStan.txt'
            else:
                print("\nThe file doesn't exist, download it from https://snap.stanford.edu/data/web-BerkStan.html")

                # if there is no folder data, create it
                if not exists('../data'):
                    os.makedirs('../data')

                # Downloading the dataset
                url = 'https://snap.stanford.edu/data/web-BerkStan.txt.gz'
                wget.download(url, '../data/web-BerkStan.txt.gz')
                # Unzipping the dataset
                with gzip.open('../data/web-BerkStan.txt.gz', 'rb') as f_in:
                    with open('../data/web-BerkStan.txt', 'wb') as f_out:
                        f_out.write(f_in.read())

                # delete the zipped file
                os.remove('../data/web-BerkStan.txt.gz')

                dataset = '../data/web-BerkStan.txt'
                print("\nDataset downloaded\n")

        return dataset

    # Creating the graph from the dataset
    def create_graph(dataset):
        print("\nCreating the graph...")
        G = nx.read_edgelist(dataset, create_using=nx.DiGraph(), nodetype=int)
        n = G.number_of_nodes()
        print("Graph created based on the dataset\n")
        return G, n

    # The matrix is filled with zeros and the (i,j) element is x if the node i is connected to the node j. Where x is 1/(number of nodes connected to i).
    def create_matrix(G):
        print("Creating the transition probability matrix...")
        P = sp.sparse.lil_matrix((n,n))
        for i in G.nodes():
            for j in G[i]: #G[i] is the list of nodes connected to i, it's neighbors
                P[i-1,j-1] = 1/len(G[i])
        print("Transition probability matrix created\n")
        return P

    # The vector is filled with d(i) = 1 if the i row of the matrix P is filled with zeros, other wise is 0
    def dangling_nodes(P,n):
        print("Creating the list of dangling nodes...")
        d = sp.sparse.lil_matrix((n,1))
        for i in range(n):
            if P[i].sum() == 0:
                d[i] = 1
        print("List of dangling nodes created\n")
        return d

    # For now it is set to equally distribute the probability to all the nodes
    def probability_vector(n):
        print("Creating the probability vector...")
        v = sp.sparse.lil_matrix((n,1))
        for i in range(n):
            v[i] = 1/n
        print("Probability vector created\n")
        return v

    def transition_matrix(P, v, d):
        print("Creating the transition matrix...")
        Pt = P + v.dot(d.T)
        print("Transition matrix created\n")
        return Pt

    # it can vary from 0 to 1, the higher the value the more the probability to jump to a random page
    def alpha():
        a = []
        for i in range(85,100):
            a.append(i/100)
        return a

# Class for plotting the results obtained. For now it can only the first algorithm. To be updated once all the algorithms are implemented and the test cases are well defined
class Plotting:
    def tau_over_iterations(df):
        x = df['tau'][::-1].tolist()
        y = df['products m-v'].tolist()

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'),
                                    layout=go.Layout(title='products needed for the convergence', xaxis_title='tau', yaxis_title='products matrix vector'))

        # save the figure as a html file
        fig.write_html("../data/results/algo1/taus_over_prods.html")
        print("The plot has been saved in the folder data/results/algo1")

    def tau_over_time(df):
        x1 = df['tau'][::-1].tolist()
        y1 = df['time'].tolist()

        fig = go.Figure(data=go.Scatter(x=x1, y=y1, mode='lines+markers'),
                                        layout=go.Layout(title='Time needed for the convergence', xaxis_title='tau', yaxis_title='time (seconds)'))

        # save the plot in a html file
        fig.write_html("../data/results/algo1/taus_over_time.html")
        print("The plot has been saved in the folder data/results/algo1")

class Algorithms:
    # Power method adapted to the PageRank problem with different damping factors. Referred as Algorithm 1 in the paper
    def algo1(Pt, v, tau, max_mv, a: list):
        start_time = time.time()

        print("STARTING ALGORITHM 1...")
        u = Pt.dot(v) - v
        mv = 1 # number of matrix-vector multiplications
        r = sp.sparse.lil_matrix((n,1))
        Res = sp.sparse.lil_matrix((len(a),1))
        x = sp.sparse.lil_matrix((n,1))

        for i in range(len(a)):
            r = a[i]*(u)
            normed_r = norm(r)
            Res[i] = normed_r

            if Res[i] > tau:
                x = r + v

        while max(Res) > tau and mv < max_mv:
            u = Pt*u
            mv += 1

            for i in range(len(a)):
                if Res[i] >= tau:
                    r = (a[i]**(mv+1))*(u)
                    Res[i] = norm(r)

                    if Res[i] > tau:
                        x = r + x

        if mv == max_mv:
            print("The algorithm didn't converge in ", max_mv, " iterations")
        else:
            print("The algorithm converged with ", mv, " matrix-vector multiplications executed")

        total_time = time.time() - start_time
        total_time = round(total_time, 2)

        print("The algorithm took ", total_time, " seconds to run\n")

        return mv, x, r, total_time

    # Refers to Algorithm 2 in the paper, it's needed to implement the algorithm 4. It doesn't work yet. Refer to the file testing.ipynb for more details. This function down here is just a place holder for now
    def Arnoldi(A, v, m):
        beta = norm(v)
        v = v/beta
        h = sp.sparse.lil_matrix((m,m))

        for j in range(1,m):
            w = A.dot(v)
            for i in range(1,j):
                h[i,j] = v.T.dot(w)
                w = w - h[i,j]*v

            h[j+i,j] = norm(w)

            if h[j+1,j] == 0:
                m = j
                v[m+1] = 0
                break
            else:
                v[j+1] = w/h[j+1,j]
        return v, h, m, beta, j

# pandas dataframe to store the results
df = pd.DataFrame(columns=['alpha', 'products m-v', 'tau', 'time'])

# Main
if __name__ == "__main__":
    dataset = Utilities.load_data()
    # maximum number of iterations, asked to the user
    max_mv = int(input("\nInsert the maximum number of iterations: "))

    G, n = Utilities.create_graph(dataset)
    P = Utilities.create_matrix(G)
    d = Utilities.dangling_nodes(P,n)
    v = Utilities.probability_vector(n)
    Pt = Utilities.transition_matrix(P, v, d)
    a = Utilities.alpha()

    # run the algorithm for different values of tau from 10^-5 to 10^-9 with step 10^-1
    for i in range(5,10):
        tau = 10**(-i)
        print("\ntau = ", tau)
        mv, x, r, total_time = Algorithms.algo1(Pt, v, tau, max_mv, a)

        # store the results in the dataframe
        df = df.append({'alpha': a, 'products m-v': mv, 'tau': tau, 'time': total_time}, ignore_index=True)

    # save the results in a csv file
    df.to_csv('../data/results/algo1/different_tau.csv', index=False)

    # plot the results
    Plotting.tau_over_iterations(df)
    Plotting.tau_over_time(df)

    # print in the terminal the columns of the dataframe iterations, tau and time
    print("Computations done. Here are the results:")
    print("\n", df[['products m-v', 'tau', 'time']])
