import FEM as FE
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math as m
import meshes as mesh
import matplotlib.pyplot as plt



def u(x,y):
  return m.cos(m.pi*2*x)*m.cos(m.pi*2*y)

def f(x,y):
  return (8*m.pi*m.pi+1)*u(x,y)



(p,t,v,z,be)=mesh.generate_gmsh(0.05,1,True)

E=mesh.edgeIndex(p,t)    

Stiff=FE.stiffnessP2(p,t,E).tocsr()
Mass=FE.massP2(p,t,E).tocsr()
Load=FE.loadP2(p,t,3,f,E)

un=spla.spsolve(Mass+Stiff,Load)

N=p[:,0].size
uc=np.zeros((N))
for j in range(N):
  uc[j]=u(p[j,0],p[j,1])



FE.plot(p,t,un[0:N])
plt.title('Diskrete Loesung')
FE.plot(p,t,un[0:N]-uc)
plt.title('Diskretisierungsfehler')
plt.show()


