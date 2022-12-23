#!/usr/bin/env python3

# Importing the libraries
import os
import wget
import gzip
import time
import scipy as sp
import numpy as np
import pandas as pd
import networkx as nx
from os.path import exists
from scipy.sparse import *
import plotly.graph_objs as go
from typing import Literal

def load_data(dataset: Literal["Stanford", "NotreDame", "BerkStan"]) -> nx.Graph:
    """Load the dataset and return a graph.

    Parameters
    ----------
    dataset : Literal["Stanford", "BerkStan"]
        The dataset to load.

    Returns
    -------
    nx.Graph
        The graph of the dataset.
        data/web-Stanford.txt

    """

    # check if there is a data folder
    if not exists(os.path.join(os.getcwd(), "data")):
        os.mkdir(os.path.join(os.getcwd(), "data"))

    if dataset not in ["Stanford", "NotreDame", "BerkStan"]:
        raise ValueError("Invalid dataset. Please choose a valid dataset.")

    # Download the dataset
    if not exists(f"../data/Web-{dataset}.txt.gz"):
        print(f"\nDownloading the dataset {dataset}...")
        wget.download(f"http://snap.stanford.edu/data/web-{dataset}.txt.gz", out=f"data/Web-{dataset}.txt.gz")
    else:
        print(f"\nThe dataset {dataset} is already downloaded.")

    # unzip the dataset
    if not exists(f"../data/Web-{dataset}.txt"):
        print(f"\nUnzipping the dataset {dataset}...")
        with gzip.open(f"../data/Web-{dataset}.txt.gz", "rb") as f_in:
            with open(f"../data/Web-{dataset}.txt", "wb") as f_out:
                f_out.write(f_in.read())

    # create the graph
    print(f"\nCreating the graph of the dataset {dataset}...\n")
    G_dataset = nx.read_edgelist(f"../data/Web-{dataset}.txt", create_using=nx.DiGraph(), nodetype=int)
    print(f"\tNumber of nodes: {G_dataset.number_of_nodes()}")
    print(f"\tNumber of edges: {G_dataset.number_of_edges()}")

    return G_dataset

def google_matrix(G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None) -> np.matrix:


    """Returns the Google matrix of the graph.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float
      The damping factor.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specfiied, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    nodelist : list, optional
      The rows and columns are ordered according to the nodes in nodelist.
      If nodelist is None, then the ordering is produced by G.nodes().

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes below). It may be common to have the dangling dict to
      be the same as the personalization dict.

    Returns
    -------
    A : NumPy matrix
       Google matrix of the graph

    Notes
    -----
    The matrix returned represents the transition matrix that describes the
    Markov chain used in PageRank. For PageRank to converge to a unique
    solution (i.e., a unique stationary distribution in a Markov chain), the
    transition matrix must be irreducible. In other words, it must be that
    there exists a path between every pair of nodes in the graph, or else there
    is the potential of "rank sinks."

    """
    if nodelist is None:
        nodelist = list(G)

    A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
    N = len(G)
    if N == 0:
        # TODO: Remove np.asmatrix wrapper in version 3.0
        return np.asmatrix(A)

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(A.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    A[dangling_nodes] = dangling_weights

    A /= A.sum(axis=1)[:, np.newaxis]  # Normalize rows to sum to 1

    return np.asmatrix(alpha * A + (1 - alpha) * p)

def google_matrix_sparse(G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None) -> np.matrix:


  """Returns the Google matrix of the graph.

  Parameters
  ----------
  G : graph
    A NetworkX graph.  Undirected graphs will be converted to a directed
    graph with two directed edges for each undirected edge.

  alpha : float
    The damping factor.

  personalization: dict, optional
    The "personalization vector" consisting of a dictionary with a
    key some subset of graph nodes and personalization value each of those.
    At least one personalization value must be non-zero.
    If not specfiied, a nodes personalization value will be zero.
    By default, a uniform distribution is used.

  nodelist : list, optional
    The rows and columns are ordered according to the nodes in nodelist.
    If nodelist is None, then the ordering is produced by G.nodes().

  weight : key, optional
    Edge data key to use as weight.  If None weights are set to 1.

  dangling: dict, optional
    The outedges to be assigned to any "dangling" nodes, i.e., nodes without
    any outedges. The dict key is the node the outedge points to and the dict
    value is the weight of that outedge. By default, dangling nodes are given
    outedges according to the personalization vector (uniform if not
    specified) This must be selected to result in an irreducible transition
    matrix (see notes below). It may be common to have the dangling dict to
    be the same as the personalization dict.

  Returns
  -------
  A : NumPy matrix
      Google matrix of the graph

  Notes
  -----
  The matrix returned represents the transition matrix that describes the
  Markov chain used in PageRank. For PageRank to converge to a unique
  solution (i.e., a unique stationary distribution in a Markov chain), the
  transition matrix must be irreducible. In other words, it must be that
  there exists a path between every pair of nodes in the graph, or else there
  is the potential of "rank sinks."

  """
  if nodelist is None:
      nodelist = list(G)

  A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format="lil", dtype=int)

  N = len(G)
  if N == 0:
      return np.asmatrix(A)

  # Personalization vector
  if personalization is None:
      p = np.repeat(1.0 / N, N)
      p = sp.sparse.lil_array(p)
  else:
      p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
      if p.sum() == 0:
          raise ZeroDivisionError
      p /= p.sum()

  # Dangling nodes
  if dangling is None:
      dangling_weights = np.ones(N, dtype=int)
      dangling_weights = sp.sparse.lil_array(dangling_weights, dtype=int)
  else:
      # Convert the dangling dictionary into an array in nodelist order
      dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
      dangling_weights /= dangling_weights.sum()

  # # Assign dangling_weights to any dangling nodes (nodes with no out links). 
  # for i in range(N):
  #   if A[[i],:].sum(axis=1) == 0:
  #     A[[i],:] = dangling_weights

  # replace rows with all zeros with dangling_weights
  A[[A.sum(axis=1)==0],:] = dangling_weights

  # Normalize rows
  row_sums = A.sum(axis=1) # row sums
  r_inv = np.power(row_sums.astype(float), -1).flatten() # inverse of row sums
  r_inv[np.isinf(r_inv)] = 0.0 # replace inf with 0
  R = sp.sparse.diags(r_inv) # create diagonal matrix
  A = R.dot(A) # normalize rows

  return A, p

def pagerank_numpy(G, alpha=0.85, personalization=None, weight="weight", dangling=None):
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specfiied, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value.


    Notes
    -----
    The eigenvector calculation uses NumPy's interface to the LAPACK
    eigenvalue solvers.  This will be the fastest and most accurate
    for small graphs.

    """
    if len(G) == 0:
        return {}
    M = google_matrix(
        G, alpha, personalization=personalization, weight=weight, dangling=dangling
    )
    # use numpy LAPACK solver
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    # eigenvector of largest eigenvalue is at ind, normalized
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = largest.sum()
    return dict(zip(G, map(float, largest / norm)))

def pagerank(G, alpha=0.85, personalization=None, max_iter=10000, tol=1.0e-9, nstart=None, weight="weight", dangling=None,):

    """
    Returns the PageRank of the nodes in the graph.

        PageRank computes a ranking of the nodes in the graph G based on
        the structure of the incoming links. It was originally designed as
        an algorithm to rank web pages.

        Parameters
        ----------
        G : graph
        A NetworkX graph.  Undirected graphs will be converted to a directed
        graph with two directed edges for each undirected edge.

        alpha : float, optional
        Damping parameter for PageRank, default=0.85.

        personalization: dict, optional
        The "personalization vector" consisting of a dictionary with a
        key some subset of graph nodes and personalization value each of those.
        At least one personalization value must be non-zero.
        If not specfiied, a nodes personalization value will 1/N where N is the
        number of nodes in G.

        max_iter : integer, optional
        Maximum number of iterations in power method eigenvalue solver.

        tol : float, optional
        Error tolerance used to check convergence in power method solver.

        nstart : dictionary, optional
        Starting value of PageRank iteration for each node.

        weight : key, optional
        Edge data key to use as weight.  If None weights are set to 1.

        dangling: dict, optional
        The outedges to be assigned to any "dangling" nodes, i.e., nodes without
        any outedges. The dict key is the node the outedge points to and the dict
        value is the weight of that outedge. By default, dangling nodes are given
        outedges according to the personalization vector (uniform if not
        specified) This must be selected to result in an irreducible transition
        matrix (see notes under google_matrix). It may be common to have the
        dangling dict to be the same as the personalization dict.

        Returns
        -------
        pagerank : dictionary
        Dictionary of nodes with PageRank as value

        Notes
        -----
        The eigenvector calculation uses power iteration with a SciPy
        sparse matrix representation.

        Raises
        ------
        PowerIterationFailedConvergence
            If the algorithm fails to converge to the specified tolerance
            within the specified number of iterations of the power iteration
            method.
    """

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1) # S[i] is the sum of the weights of edges going out of node i
    S[S != 0] = 1.0 / S[S != 0] # S[i] is now the sum of the weights of edges going into node i
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape)) # Q is the matrix of edge weights going into each node
    A = Q @ A # A is now the "stochastic matrix"

    # initial vector
    if nstart is None: # if no initial vector is specified, start with a uniform vector
        x = np.repeat(1.0 / N, N) # x is the vector of PageRank values
    else: # if an initial vector is specified, normalize it
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float) # x is the vector of PageRank values
        x /= x.sum() # normalize x

    # Personalization vector
    if personalization is None: # if no personalization vector is specified, use a uniform vector
        p = np.repeat(1.0 / N, N) # p is the personalization vector
    else: # if a personalization vector is specified, normalize it
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float) # p is the personalization vector
        if p.sum() == 0: # if the personalization vector is all zeros, use a uniform vector
            raise ZeroDivisionError
        p /= p.sum() # normalize p
    # Dangling nodes
    if dangling is None: # if no dangling nodes are specified, use a uniform vector
        dangling_weights = p # dangling_weights is the vector of dangling node weights
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float) # dangling_weights is the vector of dangling node weights
        dangling_weights /= dangling_weights.sum() # normalize dangling_weights
    is_dangling = np.where(S == 0)[0] # is_dangling is the list of dangling nodes

    # power iteration: make up to max_iter iterations
    iter = 1
    for _ in range(max_iter):
        iter += 1
        xlast = x # xlast is the previous vector of PageRank values
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p # x is the current vector of PageRank values
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum() # err is the error between the current and previous vectors of PageRank values
        print("Iteration: ", _, "\r", end="")

        if err < N * tol: # if the error is small enough, stop iterating
            return dict(zip(nodelist, map(float, x))), iter, tol # return the current vector of PageRank values'

    # other wise, return a Null dictionary, the number of iterations, and the tolerance
    # this is a failure to convergeS

    return {}, iter, tol

def shifted_pow_pagerank(G, alphas=[0.85, 0.9, 0.95, 0.99], max_iter=10000, tol=1.0e-9):

    """
    Compute the PageRank of each node in the graph G.

    Parameters
    ----------
    G : graph
        A NetworkX graph. Undirected graphs will be converted to a directed graph.

    alphas : list, optional
        A list of alpha values to use in the shifted power method. The default is [0.85, 0.9, 0.95, 0.99].

    max_iter : integer, optional
        Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
        Error tolerance used to check convergence in power method solver.

    Returns
    -------
    pagerank : dictionary
        Dictionary of nodes with PageRank as value

    mv : integer
        The number of matrix-vector multiplications used in the power method

    Notes
    -----
    The eigenvector calculation uses power iteration with a SciPy sparse matrix representation. The shifted power method is described as Algorithm 1 in the paper located in the sources folders.

    """

    N = len(G)
    if N == 0:
       raise nx.NetworkXException("Empty graph.")

    A, v = google_matrix_sparse(G)

    # initialize a sparse matrix of dimension N x len(alphas). The cols of this matrix are the page rank vectors for each alpha. Each col is v
    x = sp.sparse.lil_matrix((N, len(alphas)))
    for i in range(len(alphas)):
        # reshape v to a column vector
        v = v.reshape((N,1))
        x[:, [i]] = v

    # make v sparse
    v = sp.sparse.lil_array(v)

    mu = A @ v - v # column vector

    r = sp.sparse.lil_matrix((N, len(alphas))) # residual matrix
    res = np.ones(len(alphas)) # residual norm vector

    for i in range(len(alphas)):
        r[:,[i]] = alphas[i] * mu # residual vector for the i-th alpha
        # r_i = r[:,[i]].toarray()
        res[i] = sp.sparse.linalg.norm(r[:,[i]]) # residual norm for the i-th alpha
        if res[i] >= tol:
            x[:, [i]] = r[:,[i]] + v # update the i-th column of x
            
    mv = 1 # number of matrix-vector multiplications
    for _ in range(max_iter): # starting of the while loop in the paper
        # print("Iteration: ", _)
        err = np.max(res) 
        # print("Error: ", err)
        if err < tol:
            print("Convergence reached with, ", mv, " matrix-vector multiplications")
            return x, mv, alphas, tol

        print("Iteration: ", _, "\r", end="")


        mu = A @ mu
        mv += 1
        for i in range(len(alphas)):
            if res[i] >= tol:
                r[:,[i]] = np.power(alphas[i], mv+1) * mu
                # r_i = r[:,[i]].toarray()
                res[i] = sp.sparse.linalg.norm(r[:,[i]])

                if res[i] >= tol:
                    x[:, [i]] = r[:,[i]] + v

    raise nx.PowerIterationFailedConvergence(max_iter) # if the error is not small enough, raise an error
