import meshes as mesh
import FEM as fem
import scipy.sparse as sp
import matplotlib as plt
import numpy as np
import scipy.sparse.linalg as spla
import scipy as s
import series9_3g as ser
import math
def f(x,y):
    return (8*np.pi**2+1)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
def error(h0):
    p,t,be=mesh.grid_square(1,h0)
    h=mesh.max_mesh_width(p,t)
    e,l=mesh.edgeIndex(p,t)
    L=fem.loadP2(p,t,3,f,e,l)
    L=L.tolil()
    un=ser.main(h0,3)
    R=np.zeros(len(p))      #remove edges from right side
    for i in range(0,len(p)):
        R[i]=L[i,0]
    u1=un.transpose()*R
    u=2*np.pi**2+1/4 #das Integral ueber fu
    error=np.sqrt(abs(u-u1[0]))
    return error,h

#convergence study as in series 8
e=np.zeros(3)
ha=np.zeros(3)
for i in range(0,3): #we are only testing for 3 entries because further calculations take too long
	hi=0.1/(np.sqrt(2**i))
	er,h=error(hi)
	e[i]=er
	ha[i]=h
plt.pyplot.loglog(ha,e,'-o')
plt.pyplot.xlabel("max width h")
plt.pyplot.ylabel("error of energy norm")
plt.pyplot.show()
print("rate of convergence")
print(np.diff(np.log(e))/np.diff(np.log(ha)))