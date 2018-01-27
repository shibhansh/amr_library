from __future__ import division
import numpy as np
import numpy.matlib
import scipy.linalg
from numpy import linalg as LA


def grasshopper(W, r, _lambda=1, k=1):
    # function [l v] = grasshopper(W, r, lambda, k)
    # Reranking items by random walk on graph with absorbing states.

    # INPUT 

    #   W: n*n weight matrix with non-negative entries.
    #      W(i,j) = edge weight for the edge i->j
    #      The graph can be directed or undirected.  Self-edges are allowed.
    #   r: user-supplied initial ranking of the n items.  
    #      r is a probability vector of size n.
    #      highly ranked items have larger r values.
    #   lambda: 
    #      trade-off between relying completely on W and ignoring r (lambda=1) 
    #      and completely relying on r and ignoring W (lambda=0).
    #   k: return top k items as ranked by Grasshopper.  

    # OUTPUT 
    #   l: the top k indices after reranking.  
    #      l(1) is the best item's index (into W), l(2) the second best, and so on
    #   v: v(1) is the first item's stationary probability;
    #      v(2)..v(k) are the next k-1 item's average number of visits during
    #      absorbing random walks, in the respective iterations when they were 
    #      selected.

    # size of first element of numpy 2d array W
    # n = size(W,1)
    n = W.shape[0]

    # sanity check first
    if np.min(W)<0:
        print 'Found a negative entry in W! Stop.'
        return -1
    elif (abs(r.sum()-1)>1e-5):
        print 'The vector r does not sum to 1! Stop.'
    elif _lambda<0 or _lambda>1:
        print 'lambda is not in [0,1]! Stop.'
    elif (k<0 or k>n):
        print 'k is not in [0,n]! Stop.'
 
    # creating the graph-based transition matrix P
    row_sums = W.sum(axis=1)

    P = W / row_sums[:, numpy.newaxis]

    # incorporating user-supplied initial ranking by adding teleporting 
    hatP = _lambda*P + (1-_lambda)*np.matlib.repmat(r.transpose(), n, 1)
 
    # finding the highest ranked item from the stationary distribution
    # of hatP:  q=hatP'*q.  The item with the largest stationary probability
    # is our higest ranked item.

    q = stationary(hatP)
    # the most probably state is our first absorbing node. Put it in l.
    l = [q.argmax()]

    # reranking k-1 more items by picking out the most-visited node one by one.
    # once picked out, the item turns into an absorbing node.
    while (len(l)<k):

        # Computing the expected number of times each node will be visited 
        # before random walk is absorbed by absorbing nodes.  Averaged over
        # all start nodes.  hatP defines the transition matrix, while l specifies 
        # the absorbing nodes.

        # Compute the inverse of the fundamental matrix
        set_n = set([x for x in range(0,n)])
        set_l = set(l)
        uidx = np.array(list(set_n - set_l))
        if len(l) == 1:
            # Standard inversion if this is the first one
            N = LA.inv(np.identity(np.size(uidx)) - hatP[uidx,:][:,uidx])
        else:
            # using matrix inversion lemma henceforth
            old_uidx = np.array(list(set_n - set(l[:-1]) ))
            indx, = np.where(old_uidx==l[-1])
            N = minv(np.identity(np.size(old_uidx)) - hatP[old_uidx,:][:,old_uidx], N, indx[0])

        # Compute the expected visit counts
        nuvisit = 1/np.size(uidx) * np.dot(N.transpose(),np.ones((np.size(uidx),1)) )
        # nuvisit = N'*ones(length(uidx),1); % old version, up to scaling
        nvisit = np.zeros((n,1))
        nvisit[uidx]=nuvisit

        # Find the new absorbing state
        tmpi = nvisit.argmax()
        l.append(tmpi)

    return l

def stationary(P):
    # function q = stationary(P)
    # find the stationary distribution of the transition matrix P
    # the stationary distribution is q=P'*q
    theEigenvalues, leftEigenvectors = LA.eig(P.transpose())
    eps = 0.000001
    # Eigen vectors are along the colums of the 'leftEigenvectors'
    for index,eigen_vector in enumerate(leftEigenvectors.transpose()):
        if abs(theEigenvalues[index].real - 1.0) < eps and (theEigenvalues[index].imag - 0.0) < eps:
            required_vector = eigen_vector.real
            break
    q=abs(required_vector); # to avoid an all-negative vector
    q=q/sum(q); # make it a prob dist
    return q

def minv(A, Ainv, indx):
    # Computes the inverse of a matrix with one row and column removed using 
    # the matrix inversion lemma. It needs a matrix A, the inverse of A and the
    # row and column index which needs to be removed.

    n = A.shape[0]

    # Compute the inverse with one row removed
    u = np.zeros((n,1))
    u[indx] = -1

    v = A[indx, :]
    v[indx] = v[indx] - 1

    Ainv_u = Ainv.dot(u)
    v_Ainv = v.dot(Ainv)
    v_Ainv = v_Ainv.reshape(1, np.size(v_Ainv))

    T = Ainv - ( np.matmul(Ainv_u, v_Ainv) )/( 1+v.dot(Ainv).dot(u) )

    # Compute the inverse with one column removed
    w = A[:,indx]
    w[indx] = 0
    w = w.reshape(np.size(w),1)

    T_w = T.dot(w)
    uTranspose_T = u.transpose().dot(T)
    uTranspose_T = uTranspose_T.reshape(1, np.size(uTranspose_T))

    R = T - ( np.matmul(T_w, uTranspose_T) )/(1+u.transpose().dot(T).dot(w))

    # Remove redundant rows in resulting matrix
    R = np.delete(R,indx,0)
    R = np.delete(R,indx,1)
    return R

if __name__ == '__main__':
    W = np.array([  [0,1/3,1/3,1/3],
                    [0,0,1/2,1/2],
                    [1,0,0,0],
                    [1/2,0,1/2,0]   ])
    # W = np.array([  [0,1,0,0],
    #                 [0,0,1,0],
    #                 [0,0,0,1],
    #                 [1,1,1,1]   ])
    r = np.array([1/4,1/4,1/4,1/4])
    print grasshopper(W, r, _lambda=1, k=4)

