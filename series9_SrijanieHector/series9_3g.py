# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Series9_3g                                  ###
###            " Neumann Homogeneous                          ###
###             u(x, y)=cos(2πx)cos(2πy)                      ###
###                quadratic"                                 ###
###                                                           ###
###-----------------------------------------------------------###

# We firs import the module FEM and meshes (to generate the mesh)

import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os
import numpy.linalg as linalg

# Lets define a function that gives the solution (coefficient vector) 
# of the Neumann problem 

# input:

# h0 : Real number, maximal mesh width for the square ]0,1[^2
# f : source function
# n : the order of the numerical quadrature in the load term

def neumannP2(h0,f,n):
	# First lets generate the uniform mesh for this mesh width
	mesh=msh.grid_square(1,h0)
	# Matrix of nodes
	p=mesh[0]
	# Matrix of triangles index
	t=mesh[1]
	# Now lets compute the stiffness matrix
	Stiff=fem.stiffnessP2(p,t)
	# Now lets compute the mass matrix
	Mass=fem.massP2(p,t)
	# Lets compute the load vector
	Load=fem.loadP2(p,t,n,f)
	# The complete matrix for the bilinear form is given by the sum of 
	#the stiffness and the mass
	B=Stiff+Mass
	N=len(p)
	# Now lets get the solution of the linear system using spsolve function
	U=spla.spsolve(B,Load)
		# We return [p,t,U]
	return p,t,U[0:N]

#Now lets get the solution u=cos(2pi*x)cos(2pi*y) that gives a source
# function f(x,y)=(8pi^2+1)cos(2pi*x)cos(2pi*y)

#Lets define first the function f
f= lambda x1,x2: (8*(np.pi**2)+1)*np.cos(2*np.pi*x1)*np.cos(2*np.pi*x2)
# The closest value of h0 to 1 to be able to generate a regular 
h0=np.sqrt(2)/14
n=3

# Lets get the solution
Sol=neumannP2(h0,f,n)
p=Sol[0]
t=Sol[1]
u=Sol[2]
#Change the t to put the value 1 to zero to be able to plot with the 
# function in the FEM module
t=np.array([list(ti-1) for ti in t])

#Now lets define the exact solution as a function 
def uexact(x1,x2):
	return np.cos(2*np.pi*x1)*np.cos(2*np.pi*x2)

#Lets get a np.array of the exact solution evaluated in each 

Uexact=np.array([uexact(pi[0],pi[1]) for pi in p])

# Lets generate the plot of both
fem.plot(p,t,u,"fem_neumann.png","FEM neumann solution")
fem.plot(p,t,Uexact,"exact_neumann.png","Exact neumann solution")

# Lets get the discretization error and plot it 
error=Uexact-u
fem.plot(p,t,error,"error_neumann.png","Discretization nuemann error")


