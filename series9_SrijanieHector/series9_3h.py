# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series9_3h                                  ###
###            " Neumann Homogeneous                          ###
###             f(x,y)=(8π+1)cos(2πx)cos(2πy)                 ###
###                quadratic"                                 ###
###                                                           ###
###-----------------------------------------------------------###

# We firs import the module FEM and meshes (to generate the mesh)

import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
from scipy.optimize import curve_fit
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import neum_solverP2 as neumP2
import matplotlib.pyplot as plt



#Lets get the discretization error for the source f(x,y)=(8pi+1)cos(2pi*x)cos(2pi*y)
# which gives a analytic solution u(x,y)=((8pi+1)/(8pi^2+1))cos(2pi*x)cos(2pi*y)

#lets define the source function
f= lambda x1,x2: (8*np.pi**2+1)*np.cos(2*np.pi*x1)*np.cos(2*np.pi*x2)

# The order of the quadrature will be 3
n=3


# Lets define a function that gives the discretization error in the 
# energy norm

# We are gonna use structured uniform mesh, so it wont be necessary to
# use the max_mesh_width function since the acutal maximal width will be
# the predefined h0

# input:

# h0 : Real number, maximal mesh width for the square ]0,1[^2

def error_energyP2(h0):
	# First lets compute the discrite solution
	Sol=neumP2.neumannP2(h0,f,n)
	p=Sol[0]
	t=Sol[1]
	un=Sol[2]
	# Lets get the vector load vector
	# We need to transpose it to get do the inne producto with un
	fn=np.array([f(pi[0],pi[1]) for pi in p])
	fn=fem.load(p,t,n,f)
	fn=np.transpose(fn)[0]
	# For this source f and the analytic solution the first part of the
	# discretization error related with the load vector evaluated on 
	# analytic solution will be
	lu=1./4+2*np.pi**2
	# The part related to the approximate soluiton will be the innerproduct
	# of fn and pn
	lun=un.dot(fn)
	#finally the discretiazaion error in the energy norm
	return np.sqrt(lu-lun)

# We gonna use 6 different h0=np.sqrt(2)/n with n integer
H0=[np.sqrt(2)/i for i in [4,6,9,13,14,16,19,24]]

# We alculate the error in energy norm for this values of H0
ERROR=[error_energy(h0) for h0 in H0]

H0=np.array(H0)
ERROR=np.array(ERROR)

# We are going to plot the energy error over the
# maximal mesh width with double logarithmic scaling
ERRORlog=np.log(ERROR)
H0log=np.log(H0)

#Before plot them lets try to get a linear model of this log-log model
def linear(x,a,b):
	return a*x+b

m,b= curve_fit(linear,H0log,ERRORlog)[0]

ERRORlog_f=[linear(h0,m,b) for h0 in H0log]


# Lets plot finally the log-log scaling of the energy error
plt.plot(H0log,ERRORlog,'ro')
plt.plot(H0log,ERRORlog_f,'b')
plt.xlabel('Log(maximal width)')
plt.ylabel('Log(energy_error)')
plt.title('Energy discretization error (log-log)')
plt.savefig('energy_error.png')
plt.close()