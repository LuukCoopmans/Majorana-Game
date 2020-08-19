#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:26:04 2019
Module with functions to obtain the final overlap of the Majorana transport problem. 
@author: lcoopmans
"""
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np

def Potentialprofile(N,Vmax,sigma,xL,xR):
    """
        Returns the Potentialprofile for particular sigma, xL, xR and Vmax.
    """
    V = Vmax/(1+np.exp((np.arange(N)-xL)/sigma)) + Vmax/(1+np.exp(-(np.arange(N)-xR)/sigma))
    return V

def Kitaev_Hamiltonian(N,mu_phys,delta,t):
    """
        Returns the Kitaev_Chain Hamiltonian for open boundary conditions.
    """
    mu = (-2*t + mu_phys)*np.ones(N);
    T = t*np.ones(N-1);
    Delta = delta*np.ones(N-1);

    h = np.diag(-T,-1) + np.diag(-mu,0)+ np.diag(-T,1)
    d = np.diag(-Delta,-1) + np.diag(Delta,1)

    H = np.concatenate((np.concatenate((h,d),axis=1),np.concatenate((d.conj().T,-h.conj().T),axis=1)))
    return H

def Diagonalization(H_BdG,N):
    """
        Diagonalizes the BdG Hamiltonian and returns the W matrix and E.
    """
    D, v = np.linalg.eigh(H_BdG)

    U1 = v[0:N,N:2*N];
    V1 = v[N:2*N,N:2*N];

    W = np.concatenate((np.concatenate((U1,V1.conj()),axis=1),np.concatenate((V1,U1.conj()),axis=1)))
    E = np.diag(np.dot(np.transpose(W).conj(),np.dot(H_BdG,W)));
    return np.real(E), W


def Evolution_operator(E,W,del_t):
    """
         Returns the unitary evolution operator
    """
    U_time = np.dot(W,np.diag(np.exp(-1j*del_t*E)))
    Wi = W.T.conj()
    U_time = np.dot(U_time,Wi)
    return U_time

def Overlap(W_i,W_ev,N):
    """
        Computes the overlap between the evolved state and the initial state
        with the Onishi Formula. 
    """
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

def Transport_Majorana(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt):
    Qloss = [];

    #Construct and Diagonalize the initial BdG Hamiltonian
    V = Potentialprofile(N,Vmax[0],sigma[0],xL[0],xR);

    H_Vcom = Kitaev_Hamiltonian(N,mu_phys,Delta,t)
    H_BdG = H_Vcom + np.diag(np.concatenate((V,-V)))
    H_BdG = 1/2*(H_BdG+H_BdG.conj().T)   #make sure H is hermitian

    E, W_evolved = Diagonalization(H_BdG,N)

    for j in range(len(xL)):

        #Construct and Diagonalize the new BdG Hamiltonian
        V = Potentialprofile(N,Vmax[j],sigma[j],xL[j],xR);
        H_BdG = H_Vcom + np.diag(np.concatenate((V,-V)))
        H_BdG = 1/2*(H_BdG+H_BdG.conj().T)
        E_t, W_t = Diagonalization(H_BdG,N)

        U = Evolution_operator(E_t,W_t,dt);

        W_evolved = np.dot(U,W_evolved)
        #Qloss.append(Overlap(W_t,W_evolved,N))

    Overlap1 = Overlap(W_t,W_evolved,N)

    return Overlap1

def Transport_Majorana_all_fidelities(N,Vmax,sigma,xL,xR,mu_phys,Delta,t,dt):
    Qloss = [];

    #Construct and Diagonalize the initial BdG Hamiltonian
    V = Potentialprofile(N,Vmax[0],sigma[0],xL[0],xR);

    H_Vcom = Kitaev_Hamiltonian(N,mu_phys,Delta,t)
    H_BdG = H_Vcom + np.diag(np.concatenate((V,-V)))
    H_BdG = 1/2*(H_BdG+H_BdG.conj().T)   #make sure H is hermitian

    E, W_evolved = Diagonalization(H_BdG,N)

    for j in range(len(xL)):

        #Construct and Diagonalize the new BdG Hamiltonian
        V = Potentialprofile(N,Vmax[j],sigma[j],xL[j],xR);
        H_BdG = H_Vcom + np.diag(np.concatenate((V,-V)))
        H_BdG = 1/2*(H_BdG+H_BdG.conj().T)
        E_t, W_t = Diagonalization(H_BdG,N)

        U = Evolution_operator(E_t,W_t,dt);

        W_evolved = np.dot(U,W_evolved)
        Qloss.append(Overlap(W_t,W_evolved,N))

    Overlap1 = Overlap(W_t,W_evolved,N)

    return Qloss
