#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:50:53 2020
S-wave spin orbit coupled verison majorana transport code
@author: lcoopmans
"""
#import jax
#from jax.config import config
#config.update("jax_enable_x64", True)
#import jax.numpy as np
import numpy as np


def Potentialprofile(N,Vmax,sigma,xL,xR):
    """
        Returns the Potentialprofile for particular sigma, xL, xR and Vmax.
    """
    V = Vmax/(1+np.exp((np.arange(N)-xL)/sigma)) + Vmax/(1+np.exp(-(np.arange(N)-xR)/sigma))
    return V

def Swave_spin_orbit_Hamiltonian(N,mu_phys,delta,t,b,alpha):
    """
        Returns the Swave spin-orbit coupled Hamiltonian for open boundary conditions.
    """
    mu = (-2*t + mu_phys)*np.ones(N)
    T = t*np.ones(N-1)
    BN = b*np.ones(N)
    Alpha = alpha*np.ones(N-1)
    Delta = delta*np.ones(N)

    A1 = np.diag(-T,-1) + np.diag(-mu,0)+ np.diag(-T,1)
    A2 = np.diag(Alpha/2,-1) + np.diag(BN,0)+ np.diag(-Alpha/2,1)
    A = np.concatenate((np.concatenate((A1,A2),axis=1),np.concatenate((A2.conj().T,A1),axis=1)))

    B1 = np.zeros([N,N])
    B2 = np.diag(Delta,0)
    B = np.concatenate((np.concatenate((B1,B2),axis=1),np.concatenate((-B2,B1),axis=1)))

    H = np.concatenate((np.concatenate((A,B),axis=1),np.concatenate((B.conj().T,-A.conj().T),axis=1)))
    return H

def Diagonalization(H_BdG,N):
    """
        Diagonalizes the BdG Hamiltonian and returns the W matrix and E.
    """
    #N = np.int(np.shape(H_BdG)[0]/2)
    
    D, v = np.linalg.eigh(H_BdG)

    U1 = v[0:N,N:2*N];
    V1 = v[N:2*N,N:2*N];

    W = np.concatenate((np.concatenate((U1,V1.conj()),axis=1),np.concatenate((V1,U1.conj()),axis=1)))
    E = np.diag(np.dot(np.transpose(W).conj(),np.dot(H_BdG,W)));
    return np.real(E), W


##TODO: check conj() and inv equivalence
def Evolution_operator(E,W,del_t):
    """
         Returns the unitary evolution operator
    """
    #U_time = np.dot(np.dot(W,np.diag(np.exp(-1j*del_t*E))),np.linalg.inv(W))
    U_time = np.dot(W,np.diag(np.exp(-1j*del_t*E)))
    Wi = W.T.conj()
    U_time = np.dot(U_time,Wi)
    return U_time

def Overlap(W_i,W_ev,N):
    """
        Computes the overlap between the evolved state and the initial state
        with the Onishi Formula. 
    """
    #N = np.int(np.shape(W_i)[0]/2)
    
    U_i = W_i[0:N,0:N];
    V_i  = W_i[N:2*N,0:N];

    #switch zero modes
    P1 = np.diag(np.concatenate((np.array([0]),np.ones((N-1)))));
    P = np.concatenate((np.concatenate((P1,(np.identity(N)-P1)),axis=1),np.concatenate(((np.identity(N)-P1),P1),axis=1)));
    w_i = np.dot(W_i,P);
    U_i2 = w_i[0:N,0:N];
    V_i2  = w_i[N:2*N,0:N];

    U_ev = W_ev[0:N,0:N];
    V_ev  = W_ev[N:2*N,0:N];

    Overlap1 = np.abs(np.linalg.det((np.dot(U_i.conj().T,U_ev)+np.dot(V_i.conj().T,V_ev))));
    Overlap2 = np.abs(np.linalg.det((np.dot(U_i2.conj().T,U_ev)+np.dot(V_i2.conj().T,V_ev))));

    return np.max([Overlap1,Overlap2])

def Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,b,alpha,dt):
    Qloss = [];

    #Construct and Diagonalize the initial BdG Hamiltonian
    V = Potentialprofile(N,Vmax[0],sigma[0],xL[0],xR);

    H_Vcom = Swave_spin_orbit_Hamiltonian(N,mu_phys,Delta,t,b,alpha)
    H_BdG = H_Vcom + np.diag(np.concatenate((V,V,-V,-V)))
    H_BdG = 1/2*(H_BdG+H_BdG.conj().T)   #make sure H is hermitian

    E, W_evolved = Diagonalization(H_BdG,N+N)

    for j in range(len(xL)):
        #Construct and Diagonalize the new BdG Hamiltonian
        V = Potentialprofile(N,Vmax[j],sigma[j],xL[j],xR);
        H_BdG = H_Vcom + np.diag(np.concatenate((V,V,-V,-V)))
        H_BdG = 1/2*(H_BdG+H_BdG.conj().T)
        E_t, W_t = Diagonalization(H_BdG,N+N)

        U = Evolution_operator(E_t,W_t,dt);

        W_evolved = np.dot(U,W_evolved)
        #Qloss.append(Overlap(W_t,W_evolved))

    Overlap1 = Overlap(W_t,W_evolved,N+N)

    return Overlap1

def Transport_Majorana_all_fidelities(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,b,alpha,dt):
    Qloss = [];

    #Construct and Diagonalize the initial BdG Hamiltonian
    V = Potentialprofile(N,Vmax[0],sigma[0],xL[0],xR);

    H_Vcom = Swave_spin_orbit_Hamiltonian(N,mu_phys,Delta,t,b,alpha)
    H_BdG = H_Vcom + np.diag(np.concatenate((V,V,-V,-V)))
    H_BdG = 1/2*(H_BdG+H_BdG.conj().T)   #make sure H is hermitian

    E, W_evolved = Diagonalization(H_BdG,N+N)

    for j in range(len(xL)):
        #Construct and Diagonalize the new BdG Hamiltonian
        V = Potentialprofile(N,Vmax[j],sigma[j],xL[j],xR);
        H_BdG = H_Vcom + np.diag(np.concatenate((V,V,-V,-V)))
        H_BdG = 1/2*(H_BdG+H_BdG.conj().T)
        E_t, W_t = Diagonalization(H_BdG,N+N)

        U = Evolution_operator(E_t,W_t,dt);

        W_evolved = np.dot(U,W_evolved)
        Qloss.append(Overlap(W_t,W_evolved))

    Overlap1 = Overlap(W_t,W_evolved,N+N)

    return Qloss

