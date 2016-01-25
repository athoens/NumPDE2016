import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import meshes as me
import FEM as fem
import matplotlib.pyplot as plt

#
#linear FE for homogeneous Neumann BVP
#

def u(x,y):
    return np.cos(2*np.pi * x)*np.cos(2*np.pi*y)

#source term function f
def f(x,y):
    res = u(x,y)*(np.pi*np.pi*8+1)
    return res

h0=0.1
n=3
mesh = me.grid_square(1,h0)
#mesh = me.read_gmsh('square.msh')

#mesh contents p = nodes, t=elements, be=boundaries edges
p = mesh[0]
t = mesh[1]
be = mesh[2]
E = me.edgeIndex(p,t)
#FEM matrizes
A = fem.stiffnessP2(p,t,E) #stiffness matrix
M = fem.massP2(p,t,E) #mass matrix
L = fem.loadP2(p,t,E,n,f) # load vector
    
#our problem is now
# (A+M)u = L
    
#solve
sol = spsla.spsolve(A+M,L)
    
#plotting
fem.subplot(plt,p,t,sol[0:(len(p))])
plt.title('P2-Solution')
rsol = np.zeros((len(p),1))
for i in range(0,len(p)):
    rsol[i] = u(p[i,0],p[i,1])
fem.subplot(plt,p,t,np.abs(rsol[:,0]-sol[0:(len(p))]))
plt.title('Error')
plt.show()