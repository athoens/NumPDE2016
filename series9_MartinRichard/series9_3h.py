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
h=0.2
n=3

error = np.zeros((6,2))
widths = np.zeros((6,1))
for j in range(0,6):
    h0 = h *(np.power(2,-j/2))
    #generate uniform grid mesh with max width h0 = 0.1
    mesh = me.grid_square(1,h0)
    #mesh = me.read_gmsh('square.msh')
    
    #mesh contents p = nodes, t=elements, be=boundaries edges
    p = mesh[0]
    t = mesh[1]
    be = mesh[2]
    E = me.edgeIndex(p,t)
    #FEM matrizes
    A = fem.stiffness(p,t) #stiffness matrix
    M = fem.mass(p,t) #mass matrix
    L = fem.load(p,t,n,f) # load vector
    
    AP2 = fem.stiffnessP2(p,t,E) #stiffness matrix
    MP2 = fem.massP2(p,t,E) #mass matrix
    LP2 = fem.loadP2(p,t,E,n,f) # load vector
    
    #our problem is now
    # (A+M)u = L
    #solve
    sol = spsla.spsolve(A+M,L)
    sol = np.reshape(sol,(len(p),1))
    
    solP2 = spsla.spsolve(AP2+MP2,LP2)
    #construct u
    rsol = np.zeros((len(p),1))
    for i in range(0,len(p)):
        rsol[i] = u(p[i,0],p[i,1])
    #
    #ERROR calculation
    #
    error[j,0] = np.sqrt(np.absolute(0.25*(np.pi*np.pi*8 +1) - L.transpose().dot(sol)))
    error[j,1] = np.sqrt(np.absolute(0.25*(np.pi*np.pi*8 +1) - LP2.transpose().dot(solP2)))
    widths[j,0] = me.max_mesh_width(p,t)
#plotting
#fem.plot(p,t,sol)
plt.loglog()
plt.title('Convergence Study')
plt.xlabel('max mesh width')
plt.ylabel('error')
plt.plot(widths[:,0],error[:,0],'bo',label='P1')
plt.plot(widths[:,0],error[:,1],'r+',label='P2')
plt.legend()
plt.show()
