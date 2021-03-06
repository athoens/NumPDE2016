import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
import numpy.linalg as la
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import meshes as msh
import FEM


h0 = 0.1
qo = 1

f = lambda x,y: 0
g = lambda x,y: x+y
#u = lambda x,y: g(x,y)
[p,t]=msh.square(1,h0)

A=FEM.stiffness(p,t).tocsr()
M=FEM.mass(p,t).tocsr()
F=FEM.load(p,t,qo,f)

IN = FEM.interiorNodes(p,t)
T0 = sparse.lil_matrix((len(p),len(IN)))
for j in range(len(IN)):
  T0[IN[j],j] = 1
T0t = T0.transpose()
T0  = T0.tocsr()
T0t = T0t.tocsr()
A0 = T0t.dot(A.dot(T0))
M0 = T0t.dot(M.dot(T0))
F0 = T0t.dot(F)
Ug = np.zeros((p.shape[0]))
for i in range(0,p.shape[0]):
  Ug[i] = g(p[i,0],p[i,1])
Ug[IN] = 0
U0 = spsolve(A0+M0,F0-T0t.dot((A+M).dot(Ug)))
Un = T0.dot(U0)+Ug

FEM.plot(p,t,Un)

#U=np.zeros((p.shape[0]))
#for i in range(0,p.shape[0]):
  #U[i]=u(p[i,0],p[i,1])
#FEM.plot(p,t,U-Un)
