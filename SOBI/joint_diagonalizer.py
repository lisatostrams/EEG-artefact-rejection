# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:17:09 2017

"""

"""

"""

import numpy as np
from numpy import zeros, eye, diag
from numpy.linalg import norm

def givens_rotate( A, i, j, c, s ):
    """
    Rotate A along axis (i,j) by c and s
    """
    Ai, Aj = A[i,:], A[j,:]
    A[i,:], A[j,:] = c * Ai + s * Aj, c * Aj - s * Ai 

    return A

def givens_double_rotate( A, i, j, c, s ):
    """
    Rotate A along axis (i,j) by c and s
    """
    Ai, Aj = A[i,:], A[j,:]
    A[i,:], A[j,:] = c * Ai + s * Aj, c * Aj - s * Ai 
    A_i, A_j = A[:,i], A[:,j]
    A[:,i], A[:,j] = c * A_i + s * A_j, c * A_j - s * A_i 

    return A

def jacobi_angles( Ms, **kwargs ):
    r"""
    Simultaneously diagonalize using Jacobi angles
    @article{SC-siam,
       HTML =   "ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
       author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
       journal = "{SIAM} J. Mat. Anal. Appl.",
       title = "Jacobi angles for simultaneous diagonalization",
       pages = "161--164",
       volume = "17",
       number = "1",
       month = jan,
       year = {1995}}

    (a) Compute Givens rotations for every pair of indices (i,j) i < j 
            - from eigenvectors of G = gg'; g = A_ij - A_ji, A_ij + A_ji
            - Compute c, s as \sqrt{x+r/2r}, y/\sqrt{2r(x+r)}
    (b) Update matrices by multiplying by the givens rotation R(i,j,c,s)
    (c) Repeat (a) until stopping criterion: sin theta < threshold for all ij pairs
    """

    assert len(Ms) > 0
    print(Ms[0].shape)
    m, n = Ms[0].shape
    assert m == n

    sweeps = kwargs.get('sweeps', 500)
    threshold = kwargs.get('eps', 1e-3)
    rank = kwargs.get('rank', m)

    R = eye(m)

    for _ in range(sweeps):
        done = True
        for i in range(rank):
            for j in range(i+1, m):
                G = zeros((2,2))
                for M in Ms:
                    g = np.array([ M[i,i] - M[j,j], M[i,j] + M[j,i] ])
                    G += np.outer(g,g) / len(Ms)
                # Compute the eigenvector directly
                t_on, t_off = G[0,0] - G[1,1], G[0,1] + G[1,0]
                theta = 0.5 * np.arctan2( t_off, t_on + np.sqrt( t_on*t_on + t_off * t_off) )
                #complex angles c,s that minimize sum(off(givens_rotate(M,i,j,c,s)*M*givens_rotate(M,i,j,c,s)))
                c, s = np.cos(theta), np.sin(theta)

                if abs(s) > threshold:
                    done = False
                    # Update the matrices and V
                    for M in Ms:
                        givens_double_rotate(M, i, j, c, s)
                        #assert M[i,i] > M[j, j]
                    R = givens_rotate(R, i, j, c, s)

        if done:
            break
    R = R.T

    L = np.zeros((m, len(Ms)))
    err = 0
    for i, M in enumerate(Ms):
        # The off-diagonal elements of M should be 0
        L[:,i] = diag(M)
        err += norm(M - diag(diag(M)))

    return R, L, err

    


def fast_frobenius(Cs, **kwargs):
    """
    Ziehe (2004):
    A Fast Algorithm for Joint Diagonalization with Non-orthogonal
    Transformations and its Application to
    Blind Source Separation
    
    Input   Cs  Matrices to be diagonalized
    Output  V diagonalizer that minimizes off diagonal terms of Cs
            errs average error
    """
    K, m, n = Cs.shape
    assert m == n
    W = np.zeros([m,m])  
    _, V = np.linalg.eig(Cs[0])  #first guess for diagonalizer

    z = np.zeros([m,m])
    y = np.zeros([m,m])
    I = eye(m)
    
    Ds = np.zeros([K,m])   #diagonal terms of Cs
    Es = np.zeros_like(Cs)  #offdiagonal terms of Cs
    
    sweeps = kwargs.get('sweeps', 1000)
    theta = kwargs.get('theta', 0.5)
    
    errs = np.zeros(sweeps+1)
    
    for C in Cs:
        #calculate average initial error
        C = np.dot(np.dot(V,C),V.T)
        errs[0]+= np.linalg.norm(C - np.diag(C))/K

    for s in range(sweeps):
        
        for k in range(K):
            #set Ds and Es
            Ds[k] = np.diag(Cs[k])
            Es[k] = Cs[k]
            np.fill_diagonal(Es[k],0)
        
        #compute W from Cs according to equation 17 in article
        for i in range(m):
            for j in range(m):
                z[i,j] = sum(Ds[:,i]*Ds[:,j])
                y[i,j] = sum(Ds[:,j]*Es[:,i,j])
                
        for i in range(m):
            for j in range(m):
                W[i,j] = z[i,j]*y[j,i] - z[i,i]*y[i,j]

        #make sure W satisfies frobenius norm < theta
        if(np.linalg.norm(W,'fro') > theta):
            W = W*(theta/np.linalg.norm(W,'fro'))
            
        #update V
        V = np.dot((I + W),V)
        
        #calculate new average error
        for C in Cs:
            C = np.dot(np.dot(V,C),V.T)
            errs[s+1]+= np.linalg.norm(C - np.diag(C))/K

        if(errs[s+1] > errs[s]):
            break
        
    return V, errs[s+1]

def ACDC(Ms, **kwargs):
    """
    Yeredor (2002):
    Non-Orthogonal Joint Diagonalization in the
    Least-Squares Sense With Application in Blind
    Source Separation
    
    Input   Ms  Matrices to be diagonalized
    Output  V diagonalizer that minimizes off diagonal terms of Cs
            errs average error
    """
    sweeps = kwargs.get('sweeps', 1000)
    J, m, n = Ms.shape
    assert m == n
    e, A = np.linalg.eig(Ms[0])  #first guess for diagonalizer
    A = A.T
    err = 0
    for M in Ms:
        Mt = np.dot(np.dot(A,np.diag(np.diag(M))),A.T)
        err+= (np.linalg.norm(M - Mt)**2)/J
    print(err)
    #A = np.eye(m)
    Ds = np.zeros_like(Ms)
    ws = np.ones(J)
    for j in range(J):
        Ds[j] = np.diag(np.diag(M[j])))

    
    for sweep in range(sweeps):
        for j in range(J):
            diag_ATMjA = np.diag(np.dot(np.dot(A.T,Ms[j]),A))
            ATA = np.dot(A.T,A)
            ATAxATA = np.multiply(ATA,ATA)
            if(np.linalg.det(ATAxATA) == 0):
                Ds[j] = np.diag( np.dot(np.linalg.pinv(ATAxATA),diag_ATMjA) )
            else:
                Ds[j] = np.diag( np.dot(np.linalg.inv(ATAxATA),diag_ATMjA) )
            
        A_hat = np.zeros_like(A)
        for k in range(m):
            Pk = np.zeros_like(A)
            sumj = 0
            for j in range(J):
                sumi = 0
                for i in range(m):
                    if(i!=k):
                        sumi += Ds[j][i][i]*np.dot(A.T[i],A.T[i].T)
                Pk += ws[j]*Ds[j][k][k]*(Ms[j] - sumi)
                sumj += ws[j]*Ds[j][k][k]**2
            s, V = np.linalg.eig(Pk)
            if(s[0]>0):
                A_hat.T[k] = (np.sqrt(s[0])/np.sqrt(sumj))*V[:,0]
        A = A_hat    
        err = 0
        for M in Ms:
            Mt = np.dot(np.dot(A,np.diag(np.diag(M))),A.T)
            err+= (np.linalg.norm(M - Mt)**2)/J
        print(err)


    
    
    
    
    
    